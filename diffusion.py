# diffusion.py
import torch
import os

class DiffusionModel:
    def __init__(self, model, beta_scheduler, num_timesteps=1000):
        self.model = model
        self.beta_scheduler = beta_scheduler
        self.num_timesteps = num_timesteps
        self.alpha_scheduler = 1 - self.beta_scheduler
        self.alpha_bar_scheduler = torch.cumprod(self.alpha_scheduler, dim=0)

    def add_noise(self, x_start, noise, t):
        alpha_bar_t = self.alpha_bar_scheduler[t].view(-1, 1, 1, 1)
        noisy_image = torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_image

    def sample_ddpm(self, x_shape, device='cpu'):
        """Probabilistic sampling method (DDPM-style)."""
        x_t = torch.randn(x_shape).to(device)
        stages = []

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((x_shape[0],), t, dtype=torch.long).to(device)
            noise_pred = self.model(x_t, t_tensor)

            beta_t = self.beta_scheduler[t].to(device)
            alpha_t = self.alpha_scheduler[t].to(device)
            alpha_bar_t = self.alpha_bar_scheduler[t].to(device)

            if t > 0:
                noise = torch.randn_like(x_t).to(device)
            else:
                noise = torch.zeros_like(x_t)

            x_t = (
                1 / torch.sqrt(alpha_t)
            ) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise

            if t % (self.num_timesteps // 10) == 0 or t == 0:
                stages.append((x_t.clone().detach().cpu(), t))

        return stages
    
    def sample_ddim(self, x_shape, num_ddim_steps=50, device='cpu'):
        """Deterministic sampling method (DDIM-style) with adjustable steps."""
        x_t = torch.randn(x_shape).to(device)
        stages = []

        # Create a list of timesteps to use for sampling and reverse it
        timesteps = torch.linspace(
            0, self.num_timesteps - 1, num_ddim_steps, dtype=torch.float32
        ).long().to(device)
        timesteps = timesteps.flip(0)  # Reverse the timesteps

        for idx, t in enumerate(timesteps):
            t = t.item()
            t_tensor = torch.full((x_shape[0],), t, dtype=torch.long).to(device)
            noise_pred = self.model(x_t, t_tensor)

            alpha_t = self.alpha_scheduler[t].to(device)
            alpha_bar_t = self.alpha_bar_scheduler[t].to(device)
            
            if idx + 1 < len(timesteps):
                t_prev = timesteps[idx + 1].item()
                alpha_bar_t_prev = self.alpha_bar_scheduler[t_prev].to(device)
            else:
                # At the last timestep, set alpha_bar_t_prev to 1.0
                alpha_bar_t_prev = torch.tensor(1.0).to(device)

            # Predict x0 from the current x_t and the model's noise prediction
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            # Compute the next x_t using the DDIM update rule
            x_t = (
                torch.sqrt(alpha_bar_t_prev) * x0_pred
                + torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
            )

            # Store intermediate stages
            if idx % max(1, num_ddim_steps // 10) == 0 or t == 0:
                stages.append((x_t.clone().detach().cpu(), t))

        return stages





