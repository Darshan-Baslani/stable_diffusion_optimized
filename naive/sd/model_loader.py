from numpy import diff
import torch
from torchao import quantize_
from torchao.quantization.quant_api import int8_weight_only

from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    for k in list(state_dict['encoder'].keys()):
        if k not in encoder.state_dict():
            print("Dropping unexpected encoder key:", k)
            state_dict['encoder'].pop(k)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    for k in list(state_dict['decoder'].keys()):
        if k not in decoder.state_dict():
            print("Dropping unexpected decoder key:", k)
            state_dict['decoder'].pop(k)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device).to(torch.bfloat16)
    for k in list(state_dict['diffusion'].keys()):
        if k not in diffusion.state_dict():
            print("Dropping unexpected diffusion key:", k)
            state_dict['diffusion'].pop(k)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)
    quantize_(diffusion, int8_weight_only())

    clip = CLIP().to(device)
    for k in list(state_dict['clip'].keys()):
        if k not in clip.state_dict():
            print("Dropping unexpected clip key:", k)
            state_dict["clip"].pop(k)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }
