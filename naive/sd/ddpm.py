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

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_training_steps // self.num_inference_steps
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_t_bar = self.alpha_t_bar[timestep]
        alpha_t_bar_prev = self.alpha_t_bar[prev_t] if prev_t >= 0 else self.one

        # eq 7 of ddpm paper says we require current_beta_t for calculating var
        # one way of calculating current_beta_t is by following method which
        # is maybe not mentioned in the paper, but it is valid
        current_beta_t = 1 - alpha_t_bar / alpha_t_bar_prev

        # eq 7
        variance = (1 - alpha_t_bar_prev) / (1 - alpha_t_bar) * current_beta_t

        return variance 

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        prev_timestep = self._get_previous_timestep(timestep)

        # compute alphas and betas
        alpha_t_bar = self.alpha_t_bar[timestep]
        alpha_t_bar_prev = self.alpha_t_bar[prev_timestep] if timestep >= 0 else self.one
        beta_t_bar = 1 = alpha_t_bar
        beta_t_bar_prev = 1 - alpha_t_bar_prev
        current_alpha_t = alpha_t_bar / alpha_t_bar_prev
        current_beta_t = 1 - current_alpha_t

        # eq 15
        pred_original_sample = ((latents - (beta_t_bar ** 0.5) * model_output) / alpha_t_bar ** 0.5)

        # eq 7 for mean and variance
        first_term = ((alpha_t_bar_prev ** 0.5 * current_beta_t) / beta_t_bar) * pred_original_sample
        second_term = ((current_alpha_t ** 0.5) * beta_t_bar_prev / beta_t_bar) * latents

        mean = first_term + second_term

        variance = 0
        if timestep > 0:
            noise = torch.randn(model_output.shape, generator=self.generator, 
                                 device=model_output.device, dtype=model_output.dtype)

            stdev_noise = (self._get_variance(timestep) ** 0.5) * noise

        # sample from distribution
        pred_prev_sample = mean + stdev_noise

        return pred_prev_sample


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
