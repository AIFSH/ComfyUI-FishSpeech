import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class BaseModelArgs:
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

    # Codebook configs
    codebook_size: int = 160
    num_codebooks: int = 4
    num_in_codebooks: Optional[int] = None
    codebook_padding_idx: int = 0

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        if self.num_in_codebooks is None:
            self.num_in_codebooks = self.num_codebooks
        self.head_dim = self.dim // self.n_head


@dataclass
class NaiveModelArgs(BaseModelArgs):
    pass


@dataclass
class DualARModelArgs(BaseModelArgs):
    n_fast_layer: int = 4


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    codebook_logits: Tensor


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.config = config

        # Slow transformer
        self.embeddings = nn.Embedding(
            config.vocab_size + config.codebook_size * config.num_in_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.dim // config.n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        # For kv cache
        self.max_batch_size = -1
        self.max_seq_len = -1

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        head_dim = self.config.dim // self.config.n_head
        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                head_dim,
                dtype=dtype,
            )

    def embed(self, x: Tensor) -> Tensor:
        vocab_embeds = [self.embeddings(x[:, 0])]
        for i in range(self.config.num_in_codebooks):
            emb = self.embeddings(
                x[:, i + 1] + i * self.config.codebook_size + self.config.vocab_size
            )
            emb[x[:, i + 1] == self.config.codebook_padding_idx] = 0
            vocab_embeds.append(emb)

        x = torch.stack(vocab_embeds, dim=3)
        x = x.sum(dim=3)

        return x

    def forward(
        self, inp: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> BaseTransformerForwardResult:
        # x: (batch, num_codebooks + 1, seq_len)
        seq_len = inp.size(2)

        # Here we want to merge the embeddings of the codebooks
        x = self.embed(inp)

        mask = self.causal_mask[None, None, :seq_len, :seq_len]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[:seq_len]

        # Not that the causal mask here follows the definition of scaled_dot_product_attention
        # That is, FALSE means masked out
        # To maintain consistency, key_padding_mask use TRUE to mask out
        if key_padding_mask is not None:
            mask = mask & key_padding_mask[:, None, None, :].logical_not()

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, mask)

        # We got slow_out here
        slow_out = self.norm(x)
        token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def forward_generate(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> BaseTransformerForwardResult:
        # This is used for generation, optimized for torch compile
        assert (
            self.max_seq_len != -1 and self.max_batch_size != -1
        ), "Please call setup_caches before forward_generate"

        x = self.embed(x)

        mask = self.causal_mask[
            None, None, input_pos, : self.max_seq_len
        ]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        # If prefill, we only calculate the logits of last token
        if x.size(1) > 1:
            x = x[:, -1:]

        # We got slow_out here
        slow_out = self.norm(x)
        token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )


class NaiveTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs) -> None:
        super().__init__(config)

        self.codebook_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.codebook_output = nn.Linear(
            config.dim,
            config.codebook_size * config.num_codebooks,
            bias=False,
        )

    def decode(self, result: BaseTransformerForwardResult) -> TransformerForwardResult:
        token_logits = result.logits
        x = result.hidden_states

        # Codebook
        codebook_logits = self.codebook_output(self.codebook_norm(x))
        codebook_logits = rearrange(
            codebook_logits, "b n (c d) -> b n c d", c=self.config.num_codebooks
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward(
        self, inp: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> TransformerForwardResult:
        result = super().forward(inp, key_padding_mask)
        return self.decode(result)

    def forward_generate(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> TransformerForwardResult:
        result = super().forward_generate(x, input_pos)
        return self.decode(result)


class DualARTransformer(BaseTransformer):
    def __init__(self, config: DualARModelArgs) -> None:
        super().__init__(config)

        # Fast transformer
        self.fast_embeddings = nn.Embedding(
            config.codebook_size, config.dim, padding_idx=config.codebook_padding_idx
        )

        # The equivalent bs is so large that sdpa doesn't work
        self.fast_layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=False) for _ in range(config.n_fast_layer)
        )
        self.fast_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(
            config.dim,
            config.codebook_size,
            bias=False,
        )

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        super().setup_caches(max_batch_size, max_seq_len, dtype)

        head_dim = self.config.dim // self.config.n_head

        # Fast transformer
        # The max seq len here is the number of codebooks
        for b in self.fast_layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.n_local_heads,
                head_dim,
                dtype=dtype,
            )

    def forward(
        self, inp: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> TransformerForwardResult:
        parent_result = super().forward(inp, key_padding_mask)
        token_logits = parent_result.logits
        x = parent_result.hidden_states

        # Fast transformer
        fast_seq_len = self.config.num_codebooks
        fast_mask = self.causal_mask[
            None, None, :fast_seq_len, :fast_seq_len
        ]  # (B, N, Q, K)
        fast_freqs_cis = self.freqs_cis[:fast_seq_len]

        # Drop the last token and rotate left
        codebooks = inp[:, 1:-1, 1:]
        codebooks = F.pad(codebooks, (0, 1), value=self.config.codebook_padding_idx)
        codebook_embeddings = self.fast_embeddings(codebooks)
        x = torch.cat([x[:, None], codebook_embeddings], dim=1)
        b, s = x.size(0), x.size(2)
        x = rearrange(x, "b n s d -> (b s) n d")  # flatten the batch and seq_len

        # Remove padded part
        codebooks = rearrange(codebooks, "b n s -> (b s) n")
        codebook_mask = (codebooks == self.config.codebook_padding_idx).all(dim=-1)
        x_bs, x_len = x.size(0), x.size(1)
        x = x[~codebook_mask]

        for layer in self.fast_layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, fast_freqs_cis, fast_mask, use_reentrant=True)
            else:
                x = layer(x, fast_freqs_cis, fast_mask)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)
        codebook_logits = self.fast_output(fast_out)

        # Re-pad the codebook_logits
        buffer = torch.zeros(
            x_bs,
            x_len,
            codebook_logits.size(-1),
            device=codebook_logits.device,
            dtype=codebook_logits.dtype,
        )
        buffer[~codebook_mask] = codebook_logits
        codebook_logits = buffer

        assert codebook_logits.shape[1] == self.config.num_codebooks
        codebook_logits = rearrange(
            codebook_logits,
            "(b s) n d -> b s n d",
            b=b,
            s=s,
            n=self.config.num_codebooks,
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward_generate_fast(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        # Fast transformer
        x = x.view(1, 1, -1)

        fast_mask = self.causal_mask[
            None, None, input_pos, : self.config.num_codebooks
        ]  # (B, N, Q, K)
        fast_freqs_cis = self.freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis, fast_mask, input_pos=input_pos)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)  # only take the last token
        codebook_logits = self.fast_output(fast_out)

        return codebook_logits


class TransformerBlock(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attention = Attention(config, use_sdpa=use_sdpa)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            y = self.eq_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        return self.wo(y)

    def eq_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        # This is a standard scaled dot product attention
        # It's low efficient, but it doesn't raise cuda error

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight @ value


class FeedForward(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


if __name__ == "__main__":
    args = DualARModelArgs(
        max_seq_len=4096,
        vocab_size=32312,
        n_layer=12,
        n_fast_layer=4,
        n_head=12,
        dim=768,
        rope_base=10000,
        norm_eps=1e-5,
        codebook_size=128,
        num_codebooks=4,
    )

    model = DualARTransformer(args)
    model = model.cuda().bfloat16()
    print("Total params:", sum(i.numel() for i in model.parameters()) / 1024 / 1024)

    inputs = torch.randint(0, 100, (2, 5, 128)).cuda()
    key_padding_mask = torch.zeros(2, 128).bool().cuda()
    key_padding_mask[0, 2:] = True
    x1 = model(inputs, key_padding_mask=key_padding_mask)
    print(x1.token_logits.shape)
    print(x1.codebook_logits.shape)
