"""
Noise Scheduling and Sampling Module for Diffusion Models

This module implements the noise scheduling and sampling mechanisms for the diffusion process.
It provides three core functionalities:

1. Cosine Beta Schedule Generation:
   - Implements improved beta scheduling using a cosine function
   - Provides better training stability compared to linear schedules
   - Controls the rate of noise addition throughout the diffusion process

2. Forward Diffusion Sampling (q_sample):
   - Implements the forward diffusion process q(x_t|x_0)
   - Gradually adds noise to images according to the diffusion schedule
   - Maintains proper scaling using precomputed alpha values

3. Batch Noise Addition:
   - Handles batched operations for training efficiency
   - Manages device placement and tensor shapes
   - Implements proper random timestep sampling

Functions:
    cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        Generates a cosine schedule for the noise variance (beta).
        Args:
            timesteps: Total number of diffusion steps
            s: Offset parameter to prevent singularity
        Returns:
            Tensor of beta values for each timestep

    q_sample(x_0: torch.Tensor, t: torch.Tensor, 
             alpha_bar_sqrt: torch.Tensor, 
             one_minus_alpha_bar_sqrt: torch.Tensor,
             device: torch.device) -> torch.Tensor:
        Samples from q(x_t|x_0) for a batch of images.
        Args:
            x_0: Original images
            t: Timesteps for sampling
            alpha_bar_sqrt: Square root of cumulative alpha products
            one_minus_alpha_bar_sqrt: Square root of (1 - alpha_bar)
            device: Device to place tensors on
        Returns:
            Noisy versions of the input images at timestep t

    add_noise_to_images(batch: tuple,
                       alpha_bar_sqrt: torch.Tensor,
                       one_minus_alpha_bar_sqrt: torch.Tensor,
                       num_diffusion_steps: int,
                       device: torch.device = None) -> tuple:
        Adds noise to a batch of images for training.
        Args:
            batch: Tuple of (images, labels) from DataLoader
            alpha_bar_sqrt: Precomputed sqrt(alpha_bar) values
            one_minus_alpha_bar_sqrt: Precomputed sqrt(1 - alpha_bar) values
            num_diffusion_steps: Total number of diffusion steps
            device: Device to place tensors on
        Returns:
            Tuple of (original_images, noisy_images)

Implementation Notes:
    - All operations maintain numerical stability through proper scaling
    - Device management is handled automatically
    - Tensor shapes are managed for proper broadcasting
    - Random number generation is handled consistently across devices
"""

import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Generates a cosine schedule for the noise variance (beta).
    This schedule typically provides better sample quality than linear schedules.
    """
    def f(t):
        return torch.cos((t / timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
    x = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = f(x) / f(torch.tensor([0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clip(betas, 0.0001, 0.999)
    return betas

def q_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device):
    """
    Samples from q(x_t|x_0) using the reparameterization trick.
    This implements the forward diffusion process.
    """
    # Ensure that alpha_bar_sqrt and one_minus_alpha_bar_sqrt are on the correct device
    alpha_bar_sqrt = alpha_bar_sqrt.to(device)
    one_minus_alpha_bar_sqrt = one_minus_alpha_bar_sqrt.to(device)
    
    epsilon = torch.randn_like(x_0).to(device)
    alpha_bar_sqrt_t = alpha_bar_sqrt[t].view(-1, 1, 1, 1)
    one_minus_alpha_bar_sqrt_t = one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1)
    return alpha_bar_sqrt_t * x_0 + one_minus_alpha_bar_sqrt_t * epsilon

def add_noise_to_images(batch, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, num_diffusion_steps, device=None):
    """
    Adds noise to a batch of images according to the diffusion schedule.
    This function is used during training to generate noisy samples.
    """
    x_0, _ = batch  
    x_0 = x_0.to(device)
    t = torch.randint(low=0, 
                     high=num_diffusion_steps, 
                     size=(x_0.size(0),),
                     device=device).long()
    x_t = q_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device)
    
    return x_0, x_t


