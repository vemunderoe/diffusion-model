import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
batch_size = 64
learning_rate = 1e-4
epochs = 50
img_size = 28
timesteps = 1000

# Beta schedule
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

# Noise schedule
beta = linear_beta_schedule(timesteps)
alpha = 1.0 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# U-Net Architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        x = self.down1(x)
        x = nn.MaxPool2d(2)(x)
        x = self.down2(x)
        x = nn.MaxPool2d(2)(x)
        x = self.down3(x)
        x = nn.Upsample(scale_factor=2)(x)
        x = self.up3(x)
        x = nn.Upsample(scale_factor=2)(x)
        x = self.up2(x)
        x = self.final(x)
        return x

# Reverse Diffusion Process
class DiffusionModel:
    def __init__(self, model, beta_schedule, device):
        self.model = model
        self.beta = beta_schedule
        self.alpha_hat = alpha_hat.to(device)  # Move alpha_hat to device

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_t = self.alpha_hat[t].view(-1, 1, 1, 1)  # Ensure dimensions are compatible with x0
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise, noise

    def p_losses(self, x0, t):
        x_t, noise = self.add_noise(x0, t)
        predicted_noise = self.model(x_t, t)
        return nn.MSELoss()(predicted_noise, noise)

# Function to visualize the diffusion process
def visualize_diffusion_process(model, epoch, x0, timesteps, device):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    x = x0.to(device)
    
    for i, step in enumerate(torch.linspace(0, timesteps - 1, 10).long()):  # 10 steps
        x_t, _ = diffusion.add_noise(x, step)
        model.eval()
        with torch.no_grad():
            denoised = model(x_t, step)
        
        img = denoised[0].cpu().squeeze().clamp(0, 1)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis('off')
        axes[i].set_title(f"t={step.item()}")

    plt.suptitle(f"Epoch {epoch}")
    plt.show()

# Training setup
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = UNet().to(device)
diffusion = DiffusionModel(model, beta, device)  # Pass device to DiffusionModel
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        t = torch.randint(0, timesteps, (x.size(0),), device=device).long()
        loss = diffusion.p_losses(x, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

    # Visualization every few epochs
    if (epoch + 1) % 5 == 0:  # Adjust frequency as needed
        sample_data = next(iter(train_loader))[0][:8]  # Sample a few images from the dataset
        visualize_diffusion_process(model, epoch + 1, sample_data, timesteps, device)