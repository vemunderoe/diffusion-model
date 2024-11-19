# generate_samples.py
import torch
import time
from unet import UNet
from diffusion import DiffusionModel
from beta_scheduler import BetaScheduler
from torchvision.utils import make_grid, save_image
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained U-Net model
model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128).to(device)
model.load_state_dict(torch.load("checkpoints/mnist-linear-cfg/model_epoch_50.pth", map_location=device, weights_only=True))
model.eval()

# Define beta scheduler and diffusion model parameters
num_timesteps = 1000
scheduler = BetaScheduler(num_timesteps, device=device)
beta_scheduler = scheduler.get_schedule("linear")
diffusion = DiffusionModel(model, beta_scheduler, num_timesteps=num_timesteps)

# Sampling function
def sample_images(num_samples=100, batch_size=10):
    num_full_batches = num_samples // batch_size
    remainder = num_samples % batch_size
    samples = []

    with torch.no_grad():
        # Generate full batches
        for _ in range(num_full_batches):
            sample_batch = diffusion.sample(x_shape=(batch_size, 1, 32, 32), device=device)
            generated_imgs = sample_batch[-1][0]  # Get the last tensor
            generated_imgs = (generated_imgs + 1) / 2  # Normalize to [0, 1]
            generated_imgs = torch.clamp(generated_imgs, 0, 1)  # Clamp values to [0, 1]
            samples.append(generated_imgs)

        # Generate the remainder if necessary
        if remainder > 0:
            sample_batch = diffusion.sample(x_shape=(remainder, 1, 32, 32), device=device)
            generated_imgs = sample_batch[-1][0]
            generated_imgs = (generated_imgs + 1) / 2  # Normalize to [0, 1]
            generated_imgs = torch.clamp(generated_imgs, 0, 1)
            samples.append(generated_imgs)

    return torch.cat(samples, dim=0)  # Concatenate all batches into one tensor

# Function to generate and save samples using the sample_images method
def generate_and_save_samples(
    num_samples=64, batch_size=16, colons=8,
    path="generated_samples", filename="generated_samples.png", save_individual=False
):
    os.makedirs(path, exist_ok=True)
    start_time = time.time()

    # Sample images using the sample_images function
    generated_images_tensor = sample_images(num_samples=num_samples, batch_size=batch_size)

    # Save individual images if required
    if save_individual:
        for i, img in enumerate(generated_images_tensor):
            save_image(
                img,
                os.path.join(path, f"generated_image_{i + 1}.png"),
                normalize=True
            )

    # Create and save a grid of images
    grid = make_grid(generated_images_tensor, nrow=colons, normalize=True)
    save_image(grid, os.path.join(path, filename), normalize=True)

    end_time = time.time()
    print(f"Sample generation completed in {end_time - start_time:.2f} seconds.")    

def generate_and_save_denoising_samples(path="generated_samples", filename="denoising.png"):
    os.makedirs(path, exist_ok=True)
    start_time = time.time()

    with torch.no_grad():
        # Generate full batches    
        generated_imgs = []    
        samples = diffusion.sample(x_shape=(1, 1, 32, 32), device=device)
        for _, sample in enumerate(samples):            
            generated_img = sample[0]
            generated_img = (generated_img + 1) / 2
            generated_img = torch.clamp(generated_img, 0, 1)
            generated_imgs.append(generated_img)

        generated_imgs = torch.cat(generated_imgs, dim=0)
    # Create and save a grid of images
    grid = make_grid(generated_imgs, nrow=10, normalize=True)
    save_image(grid, os.path.join(path, filename), normalize=True)

    end_time = time.time()
    print(f"Sample generation completed in {end_time - start_time:.2f} seconds.")
    

# Example usage: Generate 128 images in batches of 32 with a grid layout of 16 x 8
generate_and_save_samples(num_samples=10, batch_size=10, colons=5, path="generated_samples", filename="5x2.png", save_individual=False)
generate_and_save_samples(num_samples=12, batch_size=12, colons=6, path="generated_samples", filename="6x2.png", save_individual=False)
generate_and_save_samples(num_samples=16, batch_size=16, colons=8, path="generated_samples", filename="8x2.png", save_individual=False)
generate_and_save_samples(num_samples=64, batch_size=32, colons=8, path="generated_samples", filename="8x8.png", save_individual=False)
generate_and_save_samples(num_samples=256, batch_size=32, colons=16, path="generated_samples", filename="16x16.png", save_individual=False)
generate_and_save_denoising_samples(path="generated_samples")

print("Sample generation completed and images saved.")
