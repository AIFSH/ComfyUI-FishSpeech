# ComfyUI-FishSpeech
a custom comfyui node for [fish-speech](https://github.com/fishaudio/fish-speech.git)

## Disclaimer  / 免责声明
We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.
我们不对代码库的任何非法使用承担任何责任. 请参阅您当地关于 DMCA (数字千年法案) 和其他相关法律法规.

## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
git clone https://github.com/AIFSH/ComfyUI-FishSpeech.git
cd ComfyUI-FishSpeech
pip install -r requirements.txt
```
`weights` will be downloaded from huggingface automatically! if you in china,make sure your internet attach the huggingface
or if you still struggle with huggingface, you may try follow [hf-mirror](https://hf-mirror.com/) to config your env.

you may meet 
```
ubprocess.CalledProcessError: Command '['cmake', '/tmp/pip-install-y_5f2bue/samplerate_578130ccaceb41abb26587e96f64988e', '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/tmp/pip-install-y_5f2bue/samplerate_578130ccaceb41abb26587e96f64988e/build/lib.linux-x86_64-cpython-310/', '-DPYTHON_EXECUTABLE=/usr/local/miniconda3/bin/python', '-DCMAKE_BUILD_TYPE=Release', '-DPACKAGE_VERSION_INFO=0.2.1']' returned non-zero exit status 1.
```
when install `samplerate`

try

```
pip -q install git+https://github.com/tuxu/python-samplerate.git@fix_cmake_dep
```

if 
```
"cannot import name 'weight_norm' from 'torch.nn.utils.parametrizations'
```
please update your `torch`

## Tutorial
todo

## Thanks
- [fish-speech](https://github.com/fishaudio/fish-speech.git)
