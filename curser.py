import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from noise import q_sample, add_noise_to_images

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Noise Prediction Model (U-Net inspired architecture)
class NoisePredictor(nn.Module):
    def __init__(self):
        super(NoisePredictor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Denoising Process (Reverse Diffusion)
class DenoisingProcess:
    def __init__(self, model, alpha, alpha_bar, device):
        self.model = model
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.device = device

    def p_sample(self, x_t, t):
        epsilon_pred = self.model(x_t)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1).to(self.device)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1).to(self.device)
        one_minus_alpha_bar_t = (1 - alpha_bar_t).view(-1, 1, 1, 1).to(self.device)
        return (x_t - one_minus_alpha_bar_t.sqrt() * epsilon_pred) / alpha_t.sqrt()
    

    def reverse_diffusion(self, x_t):
        for t in reversed(range(len(self.alpha))):
            x_t = self.p_sample(x_t, torch.tensor([t]).long().to(self.device))
        return x_t

# Training function
def train(model, data_loader, alpha, alpha_bar, device, learning_rate=1e-3, num_epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        for batch in data_loader:
            # Add noise directly within each batch
            x_0, x_t = add_noise_to_images(
                batch, 
                torch.sqrt(alpha_bar), 
                torch.sqrt(1.0 - alpha_bar),
                num_diffusion_steps=1000,
                device=device
            )
            
            t = torch.randint(0, len(alpha), (x_0.size(0),)).long().to(device)
            epsilon = torch.randn_like(x_0).to(device)
            
            epsilon_pred = model(x_t)
            loss = nn.MSELoss()(epsilon_pred, epsilon)  # Simplified loss calculation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear GPU memory after each batch
            del x_0, x_t, epsilon, epsilon_pred, loss
            torch.cuda.empty_cache()
        
        logging.info(f"Epoch {epoch + 1}/{num_epochs} completed.")



# Run training and denoising
def run_denoising(alpha, alpha_bar, data_loader, model, num_samples=4):
    model.eval()  # Set model to evaluation mode
    denoiser = DenoisingProcess(model, alpha, alpha_bar, device)
    
    # Get just one small batch
    batch, _ = next(iter(data_loader))
    # Take only the first few samples
    x_t = batch[:num_samples].to(device)
    
    with torch.no_grad():  # Disable gradient computation
        try:
            x_denoised = denoiser.reverse_diffusion(x_t)
            
            # Create a figure with subplots
            fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
            for i in range(num_samples):
                if num_samples == 1:
                    ax = axes
                else:
                    ax = axes[i]
                ax.imshow(x_denoised[i].squeeze().cpu().numpy(), cmap='gray')
                ax.axis('off')
            
            plt.savefig('denoised_samples.png')
            plt.close()
            
            # Clear memory
            del x_denoised
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Error during denoising: {str(e)}")
        finally:
            # Ensure memory is cleared
            del x_t
            torch.cuda.empty_cache()

    logging.info("Denoising completed and images saved.")
