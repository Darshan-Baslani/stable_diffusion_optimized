#!/bin/bash
pip install -q lightning
wget -c https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O v1-5-pruned-emaonly.ckpt
git clone https://github.com/Darshan-Baslani/stable_diffusion_optimized.git
mv /content/v1-5-pruned-emaonly.ckpt /content/sd_v1_only_code/naive/data/
cd /content/stable_diffusion_optimized
python naive/sd/inference.py
