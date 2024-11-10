import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import copy

# Hyperparameters
batch_size = 128
learning_rate = 1e-4  # Lower learning rate for stability
epochs = 10
img_size = 28
timesteps = 1000

# Cosine Beta Schedule
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

# Noise schedule
beta = linear_beta_schedule(timesteps)
alpha = 1.0 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# Timestep embedding function
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
    return emb

# Lightweight Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim, 1)
        self.key = nn.Conv2d(in_dim, in_dim, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, C, -1)
        k = self.key(x).view(B, C, -1)
        v = self.value(x).view(B, C, -1)
        
        attn = torch.bmm(q.permute(0, 2, 1), k) / (H * W)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        out = self.gamma * out + x
        return out

# U-Net Architecture with Timestep Embedding and Attention
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.time_embed = nn.Linear(128, 64)  # Timestep embedding layer

        # Downsampling layers with adjusted input channels
        self.down1 = nn.Sequential(nn.Conv2d(1 + 64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(32 + 64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(64 + 64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.down4 = nn.Sequential(nn.Conv2d(128 + 64, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        
        # Attention layer at bottleneck
        self.attn = SelfAttention(256)
        
        # Upsampling layers with adjusted input channels
        self.up4 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up3 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        
        # Final layer to reduce channels to 1
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        # Timestep embedding
        t_emb = get_timestep_embedding(t, 128).to(x.device)
        t_emb = self.time_embed(t_emb).view(-1, 64, 1, 1)
        
        # Downsampling path with concatenation of time embedding
        x1 = self.down1(torch.cat([x, t_emb.expand(-1, -1, x.shape[2], x.shape[3])], dim=1))
        x2 = F.max_pool2d(x1, 2)
        x2 = self.down2(torch.cat([x2, t_emb.expand(-1, -1, x2.shape[2], x2.shape[3])], dim=1))
        x3 = F.max_pool2d(x2, 2)
        x3 = self.down3(torch.cat([x3, t_emb.expand(-1, -1, x3.shape[2], x3.shape[3])], dim=1))
        x4 = F.max_pool2d(x3, 2)
        x4 = self.down4(torch.cat([x4, t_emb.expand(-1, -1, x4.shape[2], x4.shape[3])], dim=1))
        
        # Bottleneck with self-attention
        x4 = self.attn(x4)
        
        # Upsampling path with explicit resizing for concatenation
        x4 = F.interpolate(x4, size=x3.shape[2:])
        x3 = self.up4(torch.cat([x4, x3], dim=1))
        
        x3 = F.interpolate(x3, size=x2.shape[2:])
        x2 = self.up3(torch.cat([x3, x2], dim=1))
        
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x1 = self.up2(torch.cat([x2, x1], dim=1))
        
        return self.final(x1)

# Reverse Diffusion Process
class DiffusionModel:
    def __init__(self, model, beta_schedule, device):
        self.model = model
        self.beta = beta_schedule
        self.alpha_hat = alpha_hat.to(device)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise, noise

    def p_losses(self, x0, t):
        x_t, noise = self.add_noise(x0, t)
        predicted_noise = self.model(x_t, t)
        return nn.MSELoss()(predicted_noise, noise)

# Visualization functions
@torch.no_grad()
def visualize_noising_process(x0, timesteps, device):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    x = x0.to(device)

    for i, step in enumerate(torch.linspace(0, timesteps - 1, 10).long()):
        x_t, _ = diffusion.add_noise(x, step)
        img = x_t[0].cpu().squeeze().clamp(0, 1)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis('off')
        axes[i].set_title(f"t={step.item()}")

    plt.suptitle("Noising Process")
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/noising_process.png")
    plt.close(fig)

@torch.no_grad()
def visualize_denoising_process(model, epoch, x0, timesteps, device):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    x = x0.to(device)
    
    for i, step in enumerate(torch.linspace(0, timesteps - 1, 10).long()):
        x_t, _ = diffusion.add_noise(x, step)
        model.eval()
        
        t = torch.full((x.size(0),), step, device=device, dtype=torch.long)
        
        with torch.no_grad():
            denoised = model(x_t, t)
        
        img = denoised[0].cpu().squeeze().clamp(0, 1)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis('off')
        axes[i].set_title(f"t={step.item()}")

    plt.suptitle(f"Denoising Process - Epoch {epoch}")
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f"visualizations/epoch_{epoch}_denoising.png")
    plt.close(fig)

# Training setup
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
diffusion = DiffusionModel(model, beta, device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / 10))

print(f"Using device: {device}")

# Visualize the noising process on a sample image before training
sample_data = next(iter(train_loader))[0][:1]
visualize_noising_process(sample_data, timesteps, device)

# Training loop with stability improvements
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        t = torch.randint(0, timesteps, (x.size(0),), device=device).long()
        loss = diffusion.p_losses(x, t)

        optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        epoch_loss += loss.item()

    # Adjust learning rate
    scheduler.step()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")
    
    # Save model checkpoint and visualize denoising process
    # Visualization of denoising process every epoch        
    sample_data = next(iter(train_loader))[0][:8]
    visualize_denoising_process(model, epoch + 1, sample_data, timesteps, device)


os.makedirs("checkpoints", exist_ok=True)
torch.save(model, f"checkpoints/model_complete.pth")