"""
Main script for training and running a Denoising Diffusion Probabilistic Model (DDPM) on MNIST.

This script orchestrates the entire diffusion process by:
1. Setting up the training parameters and device configuration
2. Computing the noise schedule using a cosine beta schedule
3. Loading and preparing the MNIST dataset
4. Initializing and training the noise prediction model
5. Running the denoising process to generate samples

Key Parameters:
    num_diffusion_steps (int): Number of steps in the diffusion process (default: 1000)
    batch_size (int): Number of images processed in each training batch (default: 32)
    learning_rate (float): Learning rate for the Adam optimizer (default: 5e-4)
    num_epochs (int): Number of training epochs (default: 10)

The script follows these main steps:
1. Compute beta schedule and derived alpha values
2. Load and prepare MNIST dataset
3. Initialize noise prediction model
4. Train the model
5. Generate and visualize denoised samples
"""

import torch
import logging
from noise import cosine_beta_schedule
from curser import train, run_denoising, NoisePredictor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Set parameters
num_diffusion_steps = 1000
batch_size = 64
learning_rate = 5e-4
num_epochs = 10

# Step 2: Compute beta schedule and precompute alphas
beta_schedule = cosine_beta_schedule(num_diffusion_steps)
alpha = (1 - beta_schedule).to(device)
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

# Step 3: Load MNIST dataset with DataLoader
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Step 4: Initialize the model
model = NoisePredictor().to(device)

# Step 5: Train the model with the data loader
train(model, data_loader, alpha, alpha_bar, device, learning_rate=learning_rate, num_epochs=num_epochs)

# Step 6: Run denoising on small batches from the data loader and visualize results
run_denoising(alpha, alpha_bar, data_loader, model)
