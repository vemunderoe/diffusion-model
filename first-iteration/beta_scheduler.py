# beta_scheduler.py
import torch
import numpy as np

class BetaScheduler:
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

    def linear(self):
        betas = np.linspace(self.beta_start, self.beta_end, self.timesteps)
        return torch.tensor(betas, dtype=torch.float32, device=self.device)

    def quadratic(self):
        betas = np.linspace(self.beta_start**0.5, self.beta_end**0.5, self.timesteps) ** 2
        return torch.tensor(betas, dtype=torch.float32, device=self.device)

    def cosine(self, s=0.008):
        steps = np.arange(self.timesteps + 1)
        alphas_cumprod = np.cos(((steps / self.timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
        return torch.tensor(betas, dtype=torch.float32, device=self.device)

    def get_schedule(self, scheduler_type="linear"):
        if scheduler_type == "linear":
            return self.linear()
        elif scheduler_type == "quadratic":
            return self.quadratic()
        elif scheduler_type == "cosine":
            return self.cosine()
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")

# Example usage
if __name__ == "__main__":
    timesteps = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = BetaScheduler(timesteps, device=device)

    linear_betas = scheduler.get_schedule("linear")
    quadratic_betas = scheduler.get_schedule("quadratic")
    cosine_betas = scheduler.get_schedule("cosine")

    print("Linear Betas (Tensor):", linear_betas)
    print("Quadratic Betas (Tensor):", quadratic_betas)
    print("Cosine Betas (Tensor):", cosine_betas)
