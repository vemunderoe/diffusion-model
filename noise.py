import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Implementing beta schedule

# Linear beta schedule
# def get_beta_schedule(beta_start, beta_end, num_diffusion_steps):
#     return torch.linspace(beta_start, beta_end, num_diffusion_steps)

# Cosine beta schedule
def cosine_beta_schedule(timesteps, s=0.008):
    logging.info("Generating cosine beta schedule for %d timesteps.", timesteps)
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, min=1e-8, max=0.999)
    logging.info("Cosine beta schedule generated.")
    return betas

# beta_start = 1e-4
# beta_end = 0.02
num_diffusion_steps = 1000
logging.info("Number of diffusion steps set to: %d", num_diffusion_steps)

# beta_schedule = get_beta_schedule(beta_start, beta_end, num_diffusion_steps)
beta_schedule = cosine_beta_schedule(num_diffusion_steps)

# Implementing forward diffusion process
def compute_alpha(beta):
    return 1 - beta

def compute_alpha_bar(alpha):
    return torch.cumprod(alpha, dim=0)

# Precomputing necessary terms
beta = beta_schedule
alpha = compute_alpha(beta)
alpha_bar = compute_alpha_bar(alpha)
alpha_bar_sqrt = torch.sqrt(alpha_bar)
one_minus_alpha_bar = 1.0 - alpha_bar
one_minus_alpha_bar_sqrt = torch.sqrt(one_minus_alpha_bar)

# Load MNIST dataset from the same folder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader  # Added DataLoader for batching

transform = transforms.Compose([
    transforms.ToTensor(),
    # Optional: Normalize the data if needed
    # transforms.Normalize((0.5,), (0.5,)),  # Uncomment if you want pixel values in [-1, 1]
])

# Set 'download=False' since you already have the dataset locally
mnist_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
logging.info("Loaded MNIST dataset with %d samples.", len(mnist_dataset))

# Define batch size and create DataLoader
batch_size = 32
data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
logging.info("DataLoader created with batch size: %d", batch_size)

# Get a batch of images
for batch in data_loader:
    x_0, _ = batch  
    logging.info("Fetched a batch of images with shape: %s", x_0.shape)
    break

# Sampling noisy images at a given timestep
def q_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt):
    """
    x_0: Original image tensor
    t: Timestep tensor
    alpha_bar_sqrt: Precomputed sqrt of alpha_bar values
    one_minus_alpha_bar_sqrt: Precomputed sqrt of (1 - alpha_bar) values
    """
    epsilon = torch.randn_like(x_0)
    alpha_bar_sqrt_t = alpha_bar_sqrt[t].view(-1, 1, 1, 1)
    one_minus_alpha_bar_sqrt_t = one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1)
    return alpha_bar_sqrt_t * x_0 + one_minus_alpha_bar_sqrt_t * epsilon

# Generate random timesteps
t = torch.randint(0, num_diffusion_steps, (batch_size,)).long()
logging.info("Generated random timesteps: %s", t)

# Sample x_t
x_t = q_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt)    
logging.info("Sampled noisy images with shape: %s", x_t.shape)

# Visualize noisy images at different timesteps
def show_noisy_images(x_0, timesteps, alpha_bar_sqrt, one_minus_alpha_bar_sqrt):
    fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 5))
    for idx, t_value in enumerate(timesteps):
        t = torch.tensor([t_value] * x_0.shape[0]).long()
        x_t = q_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt)
        img = x_t[0].squeeze().detach().cpu().numpy()
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f"Timestep {t_value}")
        axes[idx].axis('off')
    plt.show()
    logging.info("Displayed noisy images for timesteps: %s", timesteps)

timesteps_to_visualize = [0, 100, 250, 500, 750, 999]
show_noisy_images(x_0, timesteps_to_visualize, alpha_bar_sqrt, one_minus_alpha_bar_sqrt)