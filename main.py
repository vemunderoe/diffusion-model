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
batch_size = 32
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
