import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from noise import q_sample

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
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(num_epochs):
        for x_0, _ in data_loader:
            x_0 = x_0.to(device)
            t = torch.randint(0, len(alpha), (x_0.size(0),)).long().to(device)
            epsilon = torch.randn_like(x_0).to(device)  # Generate epsilon separately
            x_t = q_sample(x_0, t, torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar), device)
            
            epsilon_pred = model(x_t)
            loss = criterion(epsilon_pred, epsilon)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")



# Run training and denoising
def run_denoising(alpha, alpha_bar, noisy_images, data_loader, timesteps=1000, learning_rate=1e-3, batch_size=32, num_epochs=5):
    model = NoisePredictor().to(device)
    denoiser = DenoisingProcess(model, alpha, alpha_bar, device)
    
    train(model, data_loader, alpha, alpha_bar, device, learning_rate, num_epochs)
    
    # Perform denoising
    x_denoised = denoiser.reverse_diffusion(noisy_images)
    
    # Visualize the denoised result
    plt.imshow(x_denoised[0].squeeze().detach().cpu().numpy(), cmap='gray')
    plt.title("Denoised Image")
    plt.axis('off')
    plt.savefig('denoised_images.png')
    plt.close()
    logging.info("Saved denoised images to 'denoised_images.png'.")
