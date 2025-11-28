import modal
import subprocess
from pathlib import Path

image = modal.Image.debian_slim().apt_install("wget")
app = modal.App("sd1.5-inference")
volume = modal.Volume.from_name("sd1.5_weights")
MODEL_DIR = Path("/model")

@app.function(volumes={MODEL_DIR: volume}, timeout=600, image=image)
def download_model(url: str):
    output_path = MODEL_DIR / "v1-5-pruned-emaonly.ckpt"
    
    subprocess.run(
        ["wget", url, "-O", str(output_path)], 
        check=True
    )
    print("downloaded weights successfully")

    volume.commit()

@app.local_entrypoint()
def main():
    url = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    download_model.remote(url)
