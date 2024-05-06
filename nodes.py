import os
import sys
import time
from subprocess import Popen
import folder_paths
import cuda_malloc
from srt import parse as SrtPare
from huggingface_hub import hf_hub_download

input_path = folder_paths.get_input_directory()
output_path = folder_paths.get_output_directory()
device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
fish_tmp_out = os.path.join(output_path, "fish_speech")
os.makedirs(fish_tmp_out, exist_ok=True)
parent_directory = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(parent_directory,"checkpoints")

class FishSpeech_INFER:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "prompt_audio": ("AUDIO",),
            "text":("STRING",{
                "multiline": True,
                "default": "你好，世界！"
            }),
            "prompt_text_by_srt":("SRT",{
                "multiline": True,
                "default": "a man voice"
            }),
            "text2semantic_type":(["medium","large"],{
                "default": "medium"
            }),
            "hf_token":("STRING",{
                "default": "your token"
            }),
            "num_samples":("INT", {
                "default":1
            }),
            "max_new_tokens": ("INT", {
                "default":0
            }),
            "top_p":("FLOAT",{
                "default": 0.7
            }),
            "repetition_penalty":("FLOAT",{
                "default": 1.5
            }),
            "temperature":("FLOAT",{
                "default": 0.7
            }),
            "compile":("BOOLEAN",{
                "default": False
            }),
            "seed":("INT",{
                "default": 42
            }),
            "half":("BOOLEAN",{
                "default": False
            }),
            "iterative_prompt":
                ("BOOLEAN",{
                "default": True
            }),
            "max_length":("INT",{
                "default": 2048
            }),
            "chunk_length":("INT",{
                "default": 30
            }),
        }}
    
    CATEGORY = "AIFSH_FishSpeech"
    RETURN_TYPES = ('AUDIO',)
    OUTPUT_NODE = False

    FUNCTION = "get_tts_wav"
    
    def get_tts_wav(self,prompt_audio,text,prompt_text_by_srt,text2semantic_type,hf_token,
                    num_samples,max_new_tokens,top_p,repetition_penalty,
                    temperature,compile,seed,half,iterative_prompt,max_length,
                    chunk_length):
        with open(prompt_text_by_srt, 'r', encoding="utf-8") as file:
            file_content = file.read()
        prompt_text = ' '.join([sub.content for sub in list(SrtPare(file_content))])
        filename = f"text2semantic-sft-{text2semantic_type}-v1-4k.pth"
        t2s_model_path = os.path.join(checkpoint_path, filename)
        if not os.path.isfile(t2s_model_path):
            hf_hub_download(repo_id="fishaudio/fish-speech-1",filename=filename,local_dir=checkpoint_path,token=hf_token)
        
        vq_model_path = os.path.join(checkpoint_path, "vq-gan-group-fsq-2x1024.pth")
        if not os.path.isfile(vq_model_path):
            hf_hub_download(repo_id="fishaudio/fish-speech-1",filename="vq-gan-group-fsq-2x1024.pth",local_dir=checkpoint_path,token=hf_token)
        
        tokenizer_path = os.path.join(checkpoint_path, "tokenizer.json")
        
        if not os.path.isfile(tokenizer_path):
            hf_hub_download(repo_id="fishaudio/fish-speech-1",filename="tokenizer.json",local_dir=checkpoint_path,token=hf_token)
            hf_hub_download(repo_id="fishaudio/fish-speech-1",filename="tokenizer_config.json",local_dir=checkpoint_path,token=hf_token)
            hf_hub_download(repo_id="fishaudio/fish-speech-1",filename="special_tokens_map.json",local_dir=checkpoint_path,token=hf_token)
        
        npy_path = os.path.join(fish_tmp_out, os.path.basename(prompt_audio))
        python_exec = sys.executable or "python"
        step_1 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {prompt_audio} -o {npy_path} -ckpt {vq_model_path} -d {device}"
        print("step 1 ",step_1)
        p = Popen(step_1,shell=True)
        p.wait()
        
        config_name = f"dual_ar_2_codebook_{text2semantic_type}"
        npy_path = os.path.join(fish_tmp_out, os.path.basename(prompt_audio)[:-4]+".npy")
        step_2 = f'{python_exec} {parent_directory}/tools/llama/generate.py --text "{text}" --prompt-text "{prompt_text}" \
            --prompt-tokens {npy_path} --config-name {config_name} --num-samples {num_samples} --max-new-tokens {max_new_tokens} \
                --top-p {top_p} --repetition-penalty {repetition_penalty} --temperature {temperature} \
                    --checkpoint-path {t2s_model_path} --tokenizer {checkpoint_path} {"--compile" if compile else "--no-compile"}\
                        --seed {seed} {"--half" if half else "--no-half"} {"--iterative-prompt" if iterative_prompt else "--no-iterative-prompt"} \
                            --max-length {max_length} --chunk-length {chunk_length} --output-path {fish_tmp_out}'
        print("step 2 ",step_2)
        p2 = Popen(step_2,shell=True)
        p2.wait()
        
        step_2_npy = os.path.join(fish_tmp_out,"codes_0.npy")
        out_wav_path = os.path.join(output_path,f"{time.time()}_fish_speech.wav")
        step_3 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {step_2_npy} -o {out_wav_path} -ckpt {vq_model_path} -d {device}"
        print("step 3 ",step_3)
        p3 = Popen(step_3,shell=True)
        p3.wait()
        return (out_wav_path, )

class PreViewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO",),}
                }

    CATEGORY = "AIFSH_FishSpeech"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        tmp_path = os.path.dirname(audio)
        audio_root = os.path.basename(tmp_path)
        return {"ui": {"audio":[audio_name,audio_root]}}

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "AIFSH_FishSpeech"

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)

class LoadSRT:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["srt", "txt"]]
        return {"required":
                    {"srt": (sorted(files),)},
                }

    CATEGORY = "AIFSH_FishSpeech"

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        srt_path = folder_paths.get_annotated_filepath(srt)
        return (srt_path,)