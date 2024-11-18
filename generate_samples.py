# generate_samples.py
import torch
import matplotlib.pyplot as plt
import time
from unet import UNet
from diffusion import DiffusionModel
from beta_scheduler import BetaScheduler
from utils import visualize_denoising_process
import os
from torchvision.utils import make_grid

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained U-Net model
model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128, num_classes=10).to(device)
#model = UNet(in_channels=3, out_channels=3, base_channels=64, embedding_dim=128, num_classes=10).to(device)
model.load_state_dict(torch.load("checkpoints/mnist-cfg/model.pth", map_location=device, weights_only=True))
model.eval()

# Define beta scheduler and diffusion model parameters
num_timesteps = 1000
scheduler = BetaScheduler(num_timesteps, device=device)
beta_scheduler = scheduler.get_schedule("cosine")
diffusion = DiffusionModel(model, beta_scheduler, num_timesteps=num_timesteps)

# Function to generate and save samples with timing
def generate_and_save_samples(method, num_samples=64, num_steps=200, path="generated_samples", grid_size=8):
    os.makedirs(path, exist_ok=True)
    start_time = time.time()

    generated_images = []

    for i in range(num_samples):
        with torch.no_grad():
            if method == 'ddpm':
                samples = diffusion.sample_ddpm(x_shape=(1, 1, 32, 32), device=device, labels=torch.tensor([i % 10], device=device), cfg_scale=0.0)
            elif method == 'ddim':
                samples = diffusion.sample_ddim(x_shape=(1, 1, 32, 32), device=device, num_ddim_steps=num_steps, eta=1)
            else:
                raise ValueError("Method must be 'ddpm' or 'ddim'.")

            #visualize_denoising_process(samples, path="generated_samples/ddpm", filename=f"123denoising_process_{i}.png")
            # Handle output when samples is a list or tuple and extract the final image
            if isinstance(samples, (list, tuple)):
                samples = samples[-1][0]  # Get the last tensor in the list/tuple

            generated_images.append(samples.squeeze(0))

    # Stack images and create a grid
    generated_images_tensor = torch.stack(generated_images)
    grid = make_grid(generated_images_tensor, nrow=grid_size, normalize=True)

    # Plot and save the grid
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(path, f"generated_samples_{method}.png"), bbox_inches='tight')
    plt.close()

    end_time = time.time()
    print(f"Sample generation with {method.upper()} completed in {end_time - start_time:.2f} seconds.")


# Generate 64 images with DDPM in an 8 x 8 grid
generate_and_save_samples(method='ddpm', num_samples=64, path="generated_samples/ddpm", grid_size=8)

# Generate 64 images with DDIM using 200 timesteps in an 8 x 8 grid
# generate_and_save_samples(method='ddim', num_samples=64, num_steps=200, path="generated_samples/ddim", grid_size=8)

print("Sample generation completed and images saved.")


