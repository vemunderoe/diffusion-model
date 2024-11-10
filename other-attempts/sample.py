import torch
import matplotlib.pyplot as plt
import os

# Model and diffusion parameters
img_size = 28
timesteps = 200

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the entire saved model directly
model_path = "checkpoints/model_complete.pth"
model = torch.load(model_path, map_location=device)
model.eval()  # Set model to evaluation mode

# Define the noise schedule to match training
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

beta = linear_beta_schedule(timesteps)
alpha = 1.0 - beta
alpha_hat = torch.cumprod(alpha, dim=0).to(device)

# Sampling function
def sample_images(model, timesteps, device, num_samples=8):
    model.eval()
    with torch.no_grad():
        x_t = torch.randn((num_samples, 1, img_size, img_size), device=device)
        
        # Reverse diffusion process
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x_t, t_tensor)
            
            alpha_t = alpha_hat[t]
            beta_t = beta[t]
            
            # Compute the next x_t step
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat[t])) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t += torch.sqrt(beta_t) * noise

    return x_t.cpu().clamp(0, 1)

# Generate and visualize samples
num_samples = 8
samples = sample_images(model, timesteps, device, num_samples)

# Plot the generated samples
fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))
for i in range(num_samples):
    axes[i].imshow(samples[i].squeeze(), cmap="gray")
    axes[i].axis('off')
plt.suptitle("Generated Samples")
plt.show()
