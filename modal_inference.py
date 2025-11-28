import modal
import subprocess

app = modal.App("sd1.5-inference")
image = (
    modal.Image.debian_slim()
    .run_commands("apt-get update", "apt-get install -y bash")
    .pip_install(["torch", "lightning", "torchao", "numpy", "argparse", "iprogress" , "pillow", "tqdm", "transformers"])
    .add_local_file(local_path="./modal_infer.sh", remote_path="/root/modal_infer.sh")
    .add_local_dir(
        local_path="./naive", 
        ignore=["data/v1-5-pruned-emaonly.ckpt"], 
        remote_path="/root/naive"
    )
)
volume = modal.Volume.from_name("sd1.5_weights")

@app.function(image=image, gpu="A10", volumes={"/root/weights": volume})
def infer():
    subprocess.run(["chmod", "+x", "/root/modal_infer.sh"])
    # Use full path inside the container
    result = subprocess.run(
        ["bash", "/root/modal_infer.sh", "--n_inf_steps", "50"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

