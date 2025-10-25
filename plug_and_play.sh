#!/bin/bash
pip install -q lightning
wget -c https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O v1-5-pruned-emaonly.ckpt
mv v1-5-pruned-emaonly.ckpt /content/stable_diffusion_optimized/naive/data/
python naive/sd/inference.py
