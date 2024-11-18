import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from diffusion import DiffusionModel
from beta_scheduler import BetaScheduler
from unet import UNet
from torchvision.transforms import Resize
from torchvision.utils import save_image
from utils import visualize_denoising_process
import time
import os

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load or generate real image datasets
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Required for Inception v3
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

real_dataset = datasets.CIFAR10(root='../data', train=False, transform=transform, download=True)
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
real_images = torch.cat([batch[0] for batch in real_loader])

print(f"Loaded {len(real_images)} real images.")

# Load pre-trained Inception v3 model
classifier = inception_v3(pretrained=True).to(device)
classifier.eval()

# Load the trained U-Net model
model = UNet(in_channels=3, out_channels=3, base_channels=64, embedding_dim=128).to(device)

model.load_state_dict(torch.load("checkpoints/cifar-10-linear/model_epoch_201.pth", map_location=device, weights_only=True))
model.eval()

# Define beta scheduler and diffusion model parameters
num_timesteps = 1000
scheduler = BetaScheduler(num_timesteps, device=device)
beta_scheduler = scheduler.get_schedule("linear")
diffusion = DiffusionModel(model, beta_scheduler, num_timesteps=num_timesteps)

resize_transform = Resize((299, 299))

# Adjust sample_images to generate samples in batches
def sample_images(num_samples=100, batch_size=10):
    num_full_batches = num_samples // batch_size
    remainder = num_samples % batch_size
    samples = []

    with torch.no_grad():
        # Generate full batches
        for i in range(num_full_batches):
            sample_batch = diffusion.sample(x_shape=(batch_size, 3, 32, 32), device=device)
            generated_imgs = sample_batch[-1][0]  # Extract the generated images from the diffusion process
            generated_imgs = (generated_imgs + 1) / 2  # Normalize to [0, 1]
            generated_imgs = torch.clamp(generated_imgs, 0, 1)  # Clamp to valid pixel range
            samples.extend([resize_transform(img) for img in generated_imgs])

        # Generate the remainder if necessary
        if remainder > 0:
            sample_batch = diffusion.sample(x_shape=(remainder, 3, 32, 32), device=device)
            generated_imgs = sample_batch[-1][0]
            generated_imgs = (generated_imgs + 1) / 2
            generated_imgs = torch.clamp(generated_imgs, 0, 1)
            samples.extend([resize_transform(img) for img in generated_imgs])

    return torch.stack(samples)  # Concatenate all samples into one tensor

# Function to calculate Inception Score (IS)
def inception_score(images, batch_size=32, splits=10):
    n_images = images.size(0)
    scores = []

    with torch.no_grad():
        for i in range(0, n_images, batch_size):
            batch = images[i:i + batch_size].to(device)
            batch = (batch + 1) / 2  # Normalize to [0, 1] for Inception v3
            preds = F.softmax(classifier(batch), dim=1)
            scores.append(preds)

    preds = torch.cat(scores, dim=0)
    split_scores = []

    for k in range(splits):
        part = preds[k * (n_images // splits):(k + 1) * (n_images // splits), :]
        py = torch.mean(part, dim=0)
        kl_div = part * (torch.log(part) - torch.log(py.unsqueeze(0)))
        split_scores.append(torch.exp(torch.mean(torch.sum(kl_div, dim=1))))

    return torch.mean(torch.tensor(split_scores)), torch.std(torch.tensor(split_scores))

# Function to calculate Fréchet Inception Distance (FID)
def calculate_fid(real_images, generated_images, batch_size=32):
    def get_activations(images):
        activations = []
        with torch.no_grad():
            for i in range(0, images.size(0), batch_size):
                batch = images[i:i + batch_size].to(device)
                batch = (batch + 1) / 2  # Normalize to [0, 1]
                pred = classifier(batch)
                activations.append(pred.cpu().numpy())
        return np.concatenate(activations, axis=0)

    act_real = get_activations(real_images)
    act_gen = get_activations(generated_images)

    mu_real, sigma_real = np.mean(act_real, axis=0), np.cov(act_real, rowvar=False)
    mu_gen, sigma_gen = np.mean(act_gen, axis=0), np.cov(act_gen, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid_score

os.makedirs("isfid", exist_ok=True)

# Sample generated images and compute metrics
num_samples = 10  # Adjust as needed
batch_size = 10  # Adjust as needed
print(f"Starting to sample {num_samples} images in batches of {batch_size}...")
print(f"Time started: {time.ctime()}")
start_time = time.time()
generated_images = sample_images(num_samples=num_samples, batch_size=batch_size)
end_time = time.time()
print(f"Time taken to sample images: {end_time - start_time:.2f} seconds")
print("Generated images shape:", generated_images.shape)  # Should print [num_samples, 1, 32, 32]

print("Generated images sampled.")

# Calculate IS and FID for CIFAR-10
is_mean, is_std = inception_score(generated_images)
print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

fid_score = calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score:.4f}")

# Save generated images
for i in range(num_samples):
    if i % 100 == 0:  # Save only 1% of the samples
        save_image(generated_images[i], f'isfid/generated_images_{i + 1}.png')    