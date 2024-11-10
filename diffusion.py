# diffusion.py
import torch
import matplotlib.pyplot as plt
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

    def sample(self, x_shape, device='cpu'):
        print("Sampling from the diffusion model...")
        x_t = torch.randn(x_shape).to(device)
        os.makedirs("visualizations/denoising_steps", exist_ok=True)

        for t in reversed(range(1, self.num_timesteps)):
            t_tensor = torch.full((x_shape[0],), t, dtype=torch.long).to(device)

            noise_pred = self.model(x_t, t_tensor)  # Pass the timestep to the model

            beta_t = self.beta_scheduler[t].to(device)
            alpha_t = self.alpha_scheduler[t].to(device)
            alpha_bar_t = self.alpha_bar_scheduler[t].to(device)

            if t > 1:
                noise = torch.randn_like(x_t).to(device)
            else:
                noise = torch.zeros_like(x_t)

            # Update x_t for the next step
            x_t = (
                1 / torch.sqrt(alpha_t)
            ) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise

            # Save intermediate images for inspection
            if t % (self.num_timesteps // 10) == 0 or t == 1:  # Save at regular intervals and final step
                plt.imshow((x_t[0].detach().cpu().squeeze() * 0.5 + 0.5).numpy(), cmap='gray')
                plt.title(f"Denoising Step {t}")
                plt.axis('off')
                plt.savefig(f"visualizations/denoising_steps/step_{t}.png")
                plt.close()

        return x_t
