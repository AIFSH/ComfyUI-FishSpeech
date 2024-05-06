import os,site,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/fish_speech.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/fish_speech\n"
                    % (now_dir,now_dir)
                )
            break
        except PermissionError:
            raise PermissionError

if os.path.isfile("%s/fish_speech.pth" % (site_packages_root)):
    print("!!!fish_speech path was added to " + "%s/fish_speech.pth" % (site_packages_root) 
    + "\n if meet `No module` error,try `python main.py` again")


WEB_DIRECTORY = "./web"
from .nodes import LoadAudio,PreViewAudio,LoadSRT,FishSpeech_INFER

NODE_CLASS_MAPPINGS = {
    "LoadAudio": LoadAudio,
    "PreViewAudio": PreViewAudio,
    "LoadSRT": LoadSRT,
    "FishSpeech_INFER": FishSpeech_INFER
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudio": "AudioLoader",
    "PreViewAudio": "PreView Audio",
    "LoadSRT": "SRT FILE Loader",
    "FishSpeech_INFER": "FishSpeech Inference"
}