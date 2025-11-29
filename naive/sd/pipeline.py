import torch
import numpy as np
from tqdm import tqdm
import logging
import time

from ddpm import DDPMSampler

HEIGHT = 512
WIDTH = 512
LATENT_HEIGHT = HEIGHT // 8
LATENT_WIDTH = WIDTH // 8

def generate(prompt: str, 
             uncond_prompt: str,
             input_image=None,
             strength=0.8,
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_inference_steps=50,
             models={},
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None,
             is_torchcompile=False,
             # do_quantize_calc=False,
             # use_quantized_weights=False,
            ):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models['clip']
        clip = clip.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if do_cfg:
                cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
                # (batch, seq_len)
                cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                cond_context = clip(cond_tokens)
                # Convert into a list of length Seq_Len=77
                uncond_tokens = tokenizer.batch_encode_plus(
                    [uncond_prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                uncond_context = clip(uncond_tokens)
                # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
                context = torch.cat([cond_context, uncond_context])
            else:
                tokens = tokenizer.batch_encode_plus(
                    [prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                context = clip(tokens)

        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"invalid sampler name: {sampler_name}")

        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device).to(torch.float16)

        diffusion = models["diffusion"].to(device)
        if is_torchcompile:
            diffusion = torch.compile(diffusion)

        timesteps = tqdm(sampler.timesteps)
        start_time = time.perf_counter()
        for i, timestep in enumerate(timesteps):
            logging.debug(f"\ninference step: {i} at timestep: {timestep}\n")
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        end_time = time.perf_counter()
        elapsed_s = end_time - start_time
        print(f"Inference time in generate func: {elapsed_s:.4f} seconds ({elapsed_s*1000:.1f} ms)")

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        latents = latents.to(torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
