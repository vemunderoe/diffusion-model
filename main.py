# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unet import UNet
from loss import ELBOLoss
from diffusion import DiffusionModel
from utils import visualize_noising_process, visualize_denoising_process
from beta_scheduler import BetaScheduler
import matplotlib.pyplot as plt
import os

# Training parameters
num_epochs = 100  # You may need to increase this for better results
batch_size = 256
learning_rate = 2e-4
num_timesteps = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Define beta scheduler for the diffusion process (e.g., linear schedule)
scheduler = BetaScheduler(num_timesteps, device=device)
beta_scheduler = scheduler.get_schedule("linear")
# beta_scheduler = scheduler.get_schedule("cosine")

# Load the MNIST dataset
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # Resize to 32x32
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
# ])
# train_dataset = datasets.MNIST(root='./', train=True, transform=transform, download=True)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are already 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels to range [-1, 1]
])
train_dataset = datasets.CIFAR10(root='./', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


# Initialize the U-Net model, diffusion model, and loss function
#model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128).to(device)
model = UNet(in_channels=3, out_channels=3, base_channels=64, embedding_dim=128).to(device)
diffusion = DiffusionModel(model, beta_scheduler, num_timesteps=num_timesteps)
#elboCriterion = ELBOLoss(beta_scheduler).to(device)
mseCriterion = torch.nn.MSELoss().to(device) # use MSE loss for simplicity

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

print(f"Started at: {os.popen('date').read()}")

#visualize_noising_process(train_dataset[0][0].unsqueeze(0), diffusion, steps=10, path="./", filename="linear_noising", image_title="Linear Noising Visualization")
visualize_noising_process(train_dataset[0][0].unsqueeze(0), diffusion, steps=10, path="./", filename="cosine_noising", image_title="Cosine Noising Visualization")

os.makedirs("checkpoints/cifar-10", exist_ok=True)

# Training loop
epoch_avg_losses = []
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
    epoch_avg_losses.append((epoch, avg_loss))

    # Visualize the denoising process at the end of each epoch
    model.eval()
    with torch.no_grad():        
        # Generate multiple samples and save them in one picture        
        #generated_samples = diffusion.sample_ddpm(x_shape=(1, 1, 32, 32), device=device)                
        generated_samples = diffusion.sample_ddpm(x_shape=(1, 3, 32, 32), device=device)                
        visualize_denoising_process(generated_samples, epoch=epoch, path="samples/generated_samples/cifar-10", filename="samples_epoch", image_title="Denoising Process Visualization")

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/cifar-10/unet_diffusion_model_checkpoint_epoch_{epoch+1}.pth")    

# Save the trained model
torch.save(model.state_dict(), "checkpoints/cifar-10/unet_diffusion_model.pth")
print("Training completed and model saved.")

# Save the epoch losses to a file
with open("epoch_losses.txt", "w") as f:
    for epoch, loss in epoch_avg_losses:
        f.write(f"Epoch {epoch + 1}: Loss {loss:.4f}\n")

print(f"Ended at: {os.popen('date').read()}")
