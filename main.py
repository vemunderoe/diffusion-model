import torch
import logging
from noise import cosine_beta_schedule, add_noise_to_images
from curser import run_denoising
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

# Set parameters
num_diffusion_steps = 1000
beta_schedule = cosine_beta_schedule(num_diffusion_steps)

# Precompute terms for noising and denoising
alpha = (1 - beta_schedule).to(device)
alpha_bar = torch.cumprod(alpha, dim=0).to(device)
alpha_bar_sqrt = torch.sqrt(alpha_bar).to(device)
one_minus_alpha_bar_sqrt = torch.sqrt(1.0 - alpha_bar).to(device)

# Add noise to images
original_images, noisy_images = add_noise_to_images(data_loader, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, num_diffusion_steps, device=device)

# Run denoising
run_denoising(alpha, alpha_bar, noisy_images, data_loader, learning_rate=5e-4, num_epochs=10)
