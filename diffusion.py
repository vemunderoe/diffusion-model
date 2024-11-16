import torch
import os

# Ensure the sampling function returns batches with consistent shapes, focusing on batch and channel dimensions

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

    def sample_ddpm(self, x_shape, device='cpu', labels=None, cfg_scale=0.0):
        """Probabilistic sampling method (DDPM-style)."""
        x_t = torch.randn(x_shape).to(device)
        stages = []

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((x_shape[0],), t, dtype=torch.long).to(device)
            
            # Pass `labels` to the model if conditional sampling is needed
            noise_pred = self.model(x_t, t_tensor, labels)
            if cfg_scale > 0:
                unconditional_noise_pred = self.model(x_t, t_tensor, None)
                noise_pred = torch.lerp(unconditional_noise_pred, noise_pred, cfg_scale)

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

            # Ensure the correct shape of the output to prevent issues downstream
            if x_t.shape != x_shape:
                raise ValueError(f"Unexpected shape after update: {x_t.shape}, expected: {x_shape}")

            if t % (self.num_timesteps // 10) == 0 or t == 0:
                stages.append((x_t.clone().detach().cpu(), t))

        return stages

    def sample_ddim(self, x_shape, device='cpu', labels=None, cfg_scale=1.0, eta=0.0, num_ddim_steps=200):
        """Deterministic sampling method (DDIM-style) as per the paper."""
        x_t = torch.randn(x_shape, device=device)
        stages = []        

        for t in reversed(range(self.num_timesteps)):            
            t_tensor = torch.full((x_shape[0],), t, dtype=torch.long, device=device)
            
            # Predict noise
            noise_pred = self.model(x_t, t_tensor, labels)
            if cfg_scale > 0:
                unconditional_noise_pred = self.model(x_t, t_tensor, None)
                noise_pred = torch.lerp(unconditional_noise_pred, noise_pred, cfg_scale)
            
            alpha_t = self.alpha_scheduler[t].to(device)
            alpha_bar_t = self.alpha_bar_scheduler[t].to(device)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            # Estimate x0
            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            
            if t > 0:
                alpha_bar_prev = self.alpha_bar_scheduler[t - 1].to(device)
                sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                sqrt_one_minus_alpha_bar_prev = torch.sqrt(1 - alpha_bar_prev)
                
                # Compute sigma_t
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                ) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
                
                # Compute the direction pointing to x0
                direction = sqrt_alpha_bar_prev * x0_pred
                
                # Compute the orthogonal component
                orthogonal = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * noise_pred
                
                # Generate noise z
                if eta > 0:
                    z = torch.randn_like(x_t, device=device)
                else:
                    z = torch.zeros_like(x_t, device=device)
                
                # Update x_t
                x_t = direction + orthogonal + sigma_t * z
            else:
                x_t = x0_pred

            # Ensure the correct shape of the output to prevent issues downstream
            if x_t.shape != x_shape:
                raise ValueError(f"Unexpected shape after update: {x_t.shape}, expected: {x_shape}")

            if t % (self.num_timesteps // 10) == 0 or t == 0:
                stages.append((x_t.clone().detach().cpu(), t))

        return stages
