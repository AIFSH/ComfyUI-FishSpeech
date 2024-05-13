import os
import sys
import time
from subprocess import Popen
import folder_paths
import cuda_malloc
import audiotsm
import audiotsm.io.wav
from pydub import AudioSegment
from srt import parse as SrtPare
from huggingface_hub import hf_hub_download

input_path = folder_paths.get_input_directory()
output_path = folder_paths.get_output_directory()
device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
fish_tmp_out = os.path.join(output_path, "fish_speech")
os.makedirs(fish_tmp_out, exist_ok=True)
parent_directory = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(parent_directory,"checkpoints")

class FishSpeech_INFER_SRT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "text":("SRT",),
            "prompt_audio": ("AUDIO",),
            "prompt_text":("SRT",),
            "if_mutiple_speaker":("BOOLEAN",{
                "default": False
            }),
            "text2semantic_type":(["medium","large"],{
                "default": "medium"
            }),
            "hf_token":("STRING",{
                "default": "your token to download weights"
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
    
    def get_tts_wav(self,text,prompt_audio,prompt_text,if_mutiple_speaker,
                    text2semantic_type,hf_token,
                    num_samples,max_new_tokens,top_p,repetition_penalty,
                    temperature,compile,seed,half,iterative_prompt,max_length,
                    chunk_length):
        
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
        
        with open(text, 'r', encoding="utf-8") as file:
            text_file_content = file.read()
        with open(prompt_text, 'r', encoding="utf-8") as file:
            prompt_text_file_content = file.read()
            
        audio_seg = AudioSegment.from_file(prompt_audio)
        prompt_subtitles = list(SrtPare(prompt_text_file_content))
        python_exec = sys.executable or "python"
        spk_aduio_dict = {}
        if if_mutiple_speaker:
            for i,prompt_sub in enumerate(prompt_subtitles):
                start_time = prompt_sub.start.total_seconds() * 1000
                end_time = prompt_sub.end.total_seconds() * 1000
                speaker = 'SPK'+prompt_sub.content[0]
                if spk_aduio_dict[speaker] is not None:
                    spk_aduio_dict[speaker] += audio_seg[start_time:end_time]
                else:
                    spk_aduio_dict[speaker] = audio_seg[start_time:end_time]
            for speaker in spk_aduio_dict.keys():
                speaker_audio_seg = spk_aduio_dict[speaker]
                speaker_audio = os.path.join(input_path, f"{speaker}.wav")
                speaker_audio_seg.export(speaker_audio,format='wav')
                npy_path = os.path.join(fish_tmp_out, os.path.basename(speaker_audio))
                step_1 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {prompt_audio} -o {npy_path} -ckpt {vq_model_path} -d {device}"
                print("step 1 ",step_1)
                p = Popen(step_1,shell=True)
                p.wait()
        else:
            npy_path = os.path.join(fish_tmp_out, "SPK0.wav")
            step_1 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {prompt_audio} -o {npy_path} -ckpt {vq_model_path} -d {device}"
            print("step 1 ",step_1)
            p = Popen(step_1,shell=True)
            p.wait()
            
        config_name = f"dual_ar_2_codebook_{text2semantic_type}"
        new_audio_seg = AudioSegment.silent(0)
        for i, (prompt_sub,text_sub) in enumerate(zip(prompt_subtitles, list(SrtPare(text_file_content)))):
            start_time = prompt_sub.start.total_seconds() * 1000
            end_time = prompt_sub.end.total_seconds() * 1000
            if i == 0:
                new_audio_seg += audio_seg[:start_time]
            
            refer_wav_seg = audio_seg[start_time:end_time]
            refer_wav = os.path.join(fish_tmp_out, f"{i}_fishspeech_refer.wav")
            refer_wav_seg.export(refer_wav, format='wav')
            
            new_text = text_sub.content
            new_prompt_text = prompt_sub.content
            if if_mutiple_speaker:
                new_text = new_text[1:]
                new_prompt_text = new_prompt_text[1:]
                npy_path = os.path.join(fish_tmp_out, f"SPK{new_text[0]}.npy")
                out_put_path = os.path.join(fish_tmp_out, f"SPK{new_text[0]}")
            else:
                npy_path = os.path.join(fish_tmp_out, "SPK0.npy")
                out_put_path = os.path.join(fish_tmp_out, "SPK0")
            os.makedirs(out_put_path, exist_ok=True)
            step_2 = f'{python_exec} {parent_directory}/tools/llama/generate.py --text "{new_text}" --prompt-text "{new_prompt_text}" \
            --prompt-tokens {npy_path} --config-name {config_name} --num-samples {num_samples} --max-new-tokens {max_new_tokens} \
                --top-p {top_p} --repetition-penalty {repetition_penalty} --temperature {temperature} \
                    --checkpoint-path {t2s_model_path} --tokenizer {checkpoint_path} {"--compile" if compile else "--no-compile"}\
                        --seed {seed} {"--half" if half else "--no-half"} {"--iterative-prompt" if iterative_prompt else "--no-iterative-prompt"} \
                            --max-length {max_length} --chunk-length {chunk_length} --output-path {out_put_path}'
            print("step 2 ",step_2)
            p2 = Popen(step_2,shell=True)
            p2.wait()
            
            step_2_npy = os.path.join(out_put_path,"codes_0.npy")
            out_wav_path = os.path.join(out_put_path,f"{i}_fish_speech.wav")
            step_3 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {step_2_npy} -o {out_wav_path} -ckpt {vq_model_path} -d {device}"
            print("step 3 ",step_3)
            p3 = Popen(step_3,shell=True)
            p3.wait()
            
            text_audio = AudioSegment.from_file(out_wav_path)
            text_audio_dur_time = text_audio.duration_seconds * 1000
            
            if i < len(prompt_subtitles) - 1:
                nxt_start = prompt_subtitles[i+1].start.total_seconds() * 1000
                dur_time =  nxt_start - start_time
            else:
                org_dur_time = audio_seg.duration_seconds * 1000
                dur_time = org_dur_time - start_time
            
            ratio = text_audio_dur_time / dur_time

            if text_audio_dur_time > dur_time:
                tmp_audio = self.map_vocal(audio=text_audio,ratio=ratio,dur_time=dur_time,
                                                wav_name=f"map_{i}_refer.wav",temp_folder=out_put_path)
                tmp_audio += AudioSegment.silent(dur_time - tmp_audio.duration_seconds*1000)
            else:
                tmp_audio = text_audio + AudioSegment.silent(dur_time - text_audio_dur_time)
          
            new_audio_seg += tmp_audio

        infer_audio = os.path.join(output_path, f"{time.time()}_fishspeech_refer.wav")
        new_audio_seg.export(infer_audio, format="wav")
        return (infer_audio, )

    def map_vocal(self,audio:AudioSegment,ratio:float,dur_time:float,wav_name:str,temp_folder:str):
        tmp_path = f"{temp_folder}/map_{wav_name}"
        audio.export(tmp_path, format="wav")
        
        clone_path = f"{temp_folder}/cloned_{wav_name}"
        reader = audiotsm.io.wav.WavReader(tmp_path)
        
        writer = audiotsm.io.wav.WavWriter(clone_path,channels=reader.channels,
                                        samplerate=reader.samplerate)
        wsloa = audiotsm.wsola(channels=reader.channels,speed=ratio)
        wsloa.run(reader=reader,writer=writer)
        audio_extended = AudioSegment.from_file(clone_path)
        return audio_extended[:dur_time]


class FishSpeech_INFER:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "prompt_audio": ("AUDIO",),
            "text":("STRING",{
                "multiline": True,
                "default": "你好啊，世界！"
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
        
        python_exec = sys.executable or "python"
        
        npy_path = os.path.join(fish_tmp_out, os.path.basename(prompt_audio))
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