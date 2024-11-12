# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unet import UNet
from loss import ELBOLoss
from diffusion import DiffusionModel
from utils import visualize_noising_process, visualize_denoising_process
import matplotlib.pyplot as plt
import os

# Training parameters
num_epochs = 100  # You may need to increase this for better results
batch_size = 128
learning_rate = 1e-4
num_timesteps = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])
train_dataset = datasets.MNIST(root='./', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define beta scheduler for the diffusion process (e.g., linear schedule)
beta_scheduler = torch.linspace(0.0001, 0.02, num_timesteps).to(device)

# Initialize the U-Net model, diffusion model, and loss function
model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128).to(device)
diffusion = DiffusionModel(model, beta_scheduler, num_timesteps=num_timesteps)
#elboCriterion = ELBOLoss(beta_scheduler).to(device)
mseCriterion = torch.nn.MSELoss().to(device) # use MSE loss for simplicity

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Started at: {os.popen('date').read()}")

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)

        # Generate noise and timesteps
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, num_timesteps, (x.size(0),), device=device)

        # Create noisy inputs
        noisy_x = diffusion.add_noise(x, noise, timesteps)

        # Predict the noise using the U-Net model (with timestep embeddings)
        predicted_noise = model(noisy_x, timesteps)

        # Compute the loss
        #loss = elboCriterion(predicted_noise, noise, timesteps)
        loss = mseCriterion(predicted_noise, noise)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Print average loss per epoch
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Visualize the denoising process at the end of each epoch
    model.eval()
    with torch.no_grad():        
        # Generate multiple samples and save them in one picture        
        generated_samples = diffusion.sample(x_shape=(1, 1, 32, 32), device=device)                
        visualize_denoising_process(generated_samples, epoch)

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/unet_diffusion_model_checkpoint_epoch_{epoch+1}.pth")    

# Save the trained model
torch.save(model.state_dict(), "checkpoints/unet_diffusion_model.pth")
print("Training completed and model saved.")

print(f"Ended at: {os.popen('date').read()}")
