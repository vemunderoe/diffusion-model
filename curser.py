import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import logging
from noise import add_noise_to_images

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, t):
        embeddings = self.time_mlp(t.float().view(-1, 1))
        return embeddings

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=None):
        super().__init__()
        self.time_dim = time_dim
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_channels)
            )

    def forward(self, x, t=None):
        x = self.conv1(x)
        if self.time_dim is not None and t is not None:
            time_emb = self.time_mlp(t)[:, :, None, None]
            x = x + time_emb
        x = self.conv2(x)
        return x

# Define the Noise Prediction Model (U-Net inspired architecture)
class NoisePredictor(nn.Module):
    def __init__(self, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Encoder
        self.inc = DoubleConv(1, 128, time_dim)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256, time_dim)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512, time_dim)
        )
        
        # Middle
        self.middle = DoubleConv(512, 512, time_dim)
        
        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256, time_dim)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128, time_dim)
        )
        
        # Final layer
        self.outc = nn.Conv2d(128, 1, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, t):
        t = self.time_embedding(t)
        
        # Encoder
        x1 = self.inc(x, t)
        x2 = self.down1[1](self.down1[0](x1), t)
        x3 = self.down2[1](self.down2[0](x2), t)
        
        # Middle
        x3 = self.middle(x3, t)
        
        # Decoder with skip connections
        x = self.up1[0](x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1[1](x, t)
        
        x = self.up2[0](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2[1](x, t)
        
        # Final layer
        x = self.dropout(x)
        return self.outc(x)

# Improved Denoising Process with Dynamic Sampling (DDIM-like)
class DenoisingProcess:
    def __init__(self, model, alpha, alpha_bar, device):
        self.model = model
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.device = device

    def p_sample(self, x_t, t, eta=0.0):
        t_tensor = torch.tensor([t]).to(self.device)
        epsilon_pred = self.model(x_t, t_tensor)
        
        alpha_t = self.alpha[t].view(-1, 1, 1, 1).to(self.device)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1).to(self.device)
        one_minus_alpha_bar_t = (1 - alpha_bar_t).view(-1, 1, 1, 1).to(self.device)

        if t > 0:
            sigma_t = eta * ((1 - alpha_t) * (1 - alpha_bar_t) / (1 - alpha_bar_t)).sqrt()
            noise = torch.randn_like(x_t)
            return (x_t - one_minus_alpha_bar_t.sqrt() * epsilon_pred) / alpha_t.sqrt() + sigma_t * noise
        else:
            return (x_t - one_minus_alpha_bar_t.sqrt() * epsilon_pred) / alpha_t.sqrt()
    
    def reverse_diffusion(self, x_t, num_steps=None, eta=0.0):
        total_steps = len(self.alpha)
        if num_steps is None:
            num_steps = total_steps
        
        step_size = max(total_steps // num_steps, 1)
        steps = range(total_steps - 1, -1, -step_size)[:num_steps]
        
        for t in steps:
            x_t = self.p_sample(x_t, torch.tensor([t]).long().to(self.device), eta=eta)
        return x_t

# Training function with mixed precision and dynamic noise scheduler
def train(model, data_loader, alpha, alpha_bar, device, learning_rate=2e-4, num_epochs=5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()
    model.train()
    denoiser = DenoisingProcess(model, alpha, alpha_bar, device)
    
    # Fixed batch for visualization
    fixed_batch, _ = next(iter(data_loader))
    fixed_batch = fixed_batch[:4].to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            x_0, x_t = add_noise_to_images(
                batch, 
                torch.sqrt(alpha_bar), 
                torch.sqrt(1.0 - alpha_bar),
                num_diffusion_steps=1000,
                device=device
            )
            
            t = torch.randint(0, len(alpha), (x_0.size(0),)).to(device)
            epsilon = torch.randn_like(x_0).to(device)
            
            with autocast():
                epsilon_pred = model(x_t, t)
                loss = F.mse_loss(epsilon_pred, epsilon)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(data_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Generate and save samples after each epoch
        model.eval()
        with torch.no_grad():
            x_t = torch.randn_like(fixed_batch).to(device)
            samples = denoiser.reverse_diffusion(x_t, eta=0.0)
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for i in range(4):
                axes[0, i].imshow(fixed_batch[i].squeeze().cpu().numpy(), cmap='gray')
                axes[0, i].axis('off')
                axes[0, i].set_title('Original')
                axes[1, i].imshow(samples[i].squeeze().cpu().numpy(), cmap='gray')
                axes[1, i].axis('off')
                axes[1, i].set_title('Generated')
            plt.tight_layout()
            plt.savefig(f'samples_epoch_{epoch+1}.png')
            plt.close()
            
            del samples
            torch.cuda.empty_cache()
        
        model.train()
        logging.info(f"Epoch {epoch + 1}/{num_epochs} completed.")

def run_denoising(alpha, alpha_bar, data_loader, model, num_samples=4):
    model.eval()
    denoiser = DenoisingProcess(model, alpha, alpha_bar, device)
    x_t = torch.randn(num_samples, 1, 28, 28).to(device)  # Assuming MNIST image size

    with torch.no_grad():
        denoising_steps = []
        num_stages = 10
        for i in range(num_stages):
            steps_for_stage = len(alpha) - int(i * len(alpha) / num_stages)
            x_denoised = denoiser.reverse_diffusion(x_t, num_steps=steps_for_stage, eta=0.0)
            denoising_steps.append(x_denoised.cpu().clone())
        
       
