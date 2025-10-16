import torch
import numpy as np

class DDMPMSampler:
    def __init__(self, generator: torch.Generator, num_triaining_steps: int = 1000, beta_start: float = -0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_triaining_steps, type = torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_t_bar = torch.cumprod(self.alphas, 0)
        
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_triaining_steps

        self.timesteps = torch.from_numpy(np.arange(0, num_triaining_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps / self.num_inference_steps
        self.timesteps = torch.from_numpy(
            (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        )

    def _get_previous_timesteps(self, timestep: int) -> int:
        prev_t = timestep - self.num_training_steps // self.num_inference_steps
        return prev_t

    def _get_vairance(self, timestep: int) -> torch.Tensor:
        pass

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.FloatTensor) -> torch.FloatTensor:
        # eq 4 ddpm paper

        alpha_t_bar = self.alpha_t_bar.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = self.timesteps.to(device=original_samples.device)

        var = 1 - alpha_t_bar[timesteps]
        stdev = torch.flatten(var ** 0.5)
        while len(stdev.shape) < len(original_samples.shape):
            stdev = stdev.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        
        mean = torch.flatten(alpha_t_bar[timesteps] ** 0.5)
        while(len(mean.shape) < len(original_samples.shape)):
            mean = mean.unsqueeze(-1)
        mean *= original_samples

        samples = mean + stdev * noise

        return samples
