import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)

# Cosine beta schedule
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, min=1e-8, max=0.999)

# Sampling noisy images at a given timestep
def q_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device):
    # Ensure that alpha_bar_sqrt and one_minus_alpha_bar_sqrt are on the correct device
    alpha_bar_sqrt = alpha_bar_sqrt.to(device)
    one_minus_alpha_bar_sqrt = one_minus_alpha_bar_sqrt.to(device)
    
    epsilon = torch.randn_like(x_0).to(device)
    alpha_bar_sqrt_t = alpha_bar_sqrt[t].view(-1, 1, 1, 1)
    one_minus_alpha_bar_sqrt_t = one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1)
    return alpha_bar_sqrt_t * x_0 + one_minus_alpha_bar_sqrt_t * epsilon


# Add noise to images and return the original and noisy images
# In noise.py
def add_noise_to_images(batch, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, num_diffusion_steps, device=None):
    x_0, _ = batch  
    x_0 = x_0.to(device)
    t = torch.randint(low=0, 
                     high=num_diffusion_steps, 
                     size=(x_0.size(0),),  # size should be a separate argument
                     device=device).long()
    x_t = q_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device)
    
    return x_0, x_t


