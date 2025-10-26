import time
import logging
import argparse

logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',
    filename='shape_log.txt',
    filemode='w',
)

import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

torch.set_num_threads(1)

tokenizer = CLIPTokenizer("naive/data/vocab.json", merges_file="naive/data/merges.txt")
model_file = "naive/data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""
do_cfg = True
cfg_scale = 8
sampler = "ddpm"
seed = 42

parser = argparse.ArgumentParser()
parser.add_argument('--n_inf_steps', type=int, default=50)
args = parser.parse_args()
num_inference_steps = args.n_inf_steps


# Start timer
start = time.perf_counter()

# If using CUDA, ensure previous kernels finished before starting timer (optional)
if DEVICE == "cuda":
    torch.cuda.synchronize()

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    # input_image=input_image,
    strength=0.8,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# If using CUDA, wait for kernels to finish before stopping timer
if DEVICE == "cuda":
    torch.cuda.synchronize()

end = time.perf_counter()
elapsed_s = end - start
print(f"Inference time: {elapsed_s:.4f} seconds ({elapsed_s*1000:.1f} ms)")

Image.fromarray(output_image).save("output.png")
