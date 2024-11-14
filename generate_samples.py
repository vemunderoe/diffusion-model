# generate_samples.py
import torch
import matplotlib.pyplot as plt
import time
from unet import UNet
from diffusion import DiffusionModel
from beta_scheduler import BetaScheduler
from utils import visualize_denoising_process
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained U-Net model
model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128).to(device)
model.load_state_dict(torch.load("checkpoints/unet_diffusion_model.pth", map_location=device, weights_only=True))
model.eval()

# Define beta scheduler and diffusion model parameters
num_timesteps = 1000
scheduler = BetaScheduler(num_timesteps, device=device)
beta_scheduler = scheduler.get_schedule("linear")
diffusion = DiffusionModel(model, beta_scheduler, num_timesteps=num_timesteps)

# Function to generate and save samples with timing
def generate_and_save_samples(method, num_samples=100, num_steps=200, path="generated_samples"):
    os.makedirs(path, exist_ok=True)
    start_time = time.time()

    for i in range(num_samples):
        with torch.no_grad():
            if method == 'ddpm':
                samples = diffusion.sample_ddpm(x_shape=(1, 1, 32, 32), device=device)
            elif method == 'ddim':
                samples = diffusion.sample_ddim(x_shape=(1, 1, 32, 32), device=device, num_ddim_steps=num_steps)
            else:
                raise ValueError("Method must be 'ddpm' or 'ddim'.")

        visualize_denoising_process(samples, path=path, filename=f"sample_{i+1}")

    end_time = time.time()
    print(f"Sample generation with {method.upper()} completed in {end_time - start_time:.2f} seconds.")

# Generate 100 images with DDPM
generate_and_save_samples(method='ddpm', num_samples=10, path="generated_samples/ddpm")

# Generate 100 images with DDIM using 200 timesteps
generate_and_save_samples(method='ddim', num_samples=10, num_steps=200, path="generated_samples/ddim")

print("Sample generation completed and images saved.")
