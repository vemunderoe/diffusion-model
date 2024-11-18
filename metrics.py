# metrics.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from torchvision import transforms, datasets
from classifier import SimpleCNN

#from diffusion import DiffusionModel
from second_dffusion import DiffusionModel

#from unet import UNet
from second_unet import UNet

from beta_scheduler import BetaScheduler
from scipy.linalg import sqrtm
from utils import visualize_denoising_process
import time

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load or generate real image datasets
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure size compatibility
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

real_dataset = datasets.MNIST(root='./', train=False, transform=transform, download=True)
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
real_images = torch.cat([batch[0] for batch in real_loader])

print(f"Loaded {len(real_images)} real images.")

# Load the pre-trained SimpleCNN model
classifier = SimpleCNN().to(device)
classifier.load_state_dict(torch.load("mnist_classifier.pth", weights_only=True))
classifier.eval()

# Load the trained U-Net model
#model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128, num_classes=None).to(device)
model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128).to(device)
model.load_state_dict(torch.load("checkpoints/mnist-new-unet-cosine/model_checkpoint_epoch_50.pth", map_location=device, weights_only=True))
model.eval()

# Define beta scheduler and diffusion model parameters
num_timesteps = 1000
scheduler = BetaScheduler(num_timesteps, device=device)
beta_scheduler = scheduler.get_schedule("cosine")
diffusion = DiffusionModel(model, beta_scheduler, num_timesteps=num_timesteps)

def sample_images_wrong(num_samples=100):
    samples = []
    for i in range(num_samples):
        with torch.no_grad():
            #sample = diffusion.sample_ddpm(x_shape=(1, 1, 32, 32), device=device, labels=torch.tensor([i % 10], device=device), cfg_scale=0.0)
            sample = diffusion.sample_ddpm(x_shape=(1, 1, 32, 32), device=device)
            samples.append(sample[-1][0].squeeze(0))  # Remove the extra dimension
            visualize_denoising_process(sample, path="isfid_gray", filename=f"denoising_process_{i + 1}.png")
            save_image(sample[-1][0].squeeze(0), f'isfid_gray/generated_images_{i + 1}_gray.png')

    return torch.stack(samples)  # This will have shape [num_samples, 1, 32, 32]


def sample_images_old(num_samples=100):
    samples = []
    for i in range(num_samples):
        with torch.no_grad():
            #sample = diffusion.sample_ddpm(x_shape=(1, 1, 32, 32), device=device, labels=torch.tensor([i % 10], device=device), cfg_scale=0.0)
            sample = diffusion.sample_ddpm(x_shape=(1, 1, 32, 32), device=device)
            generated_img = sample[-1][0].squeeze(0)  # Remove the extra dimension

            # Rescale from [-1, 1] to [0, 1] and clamp to ensure valid pixel values
            generated_img = (generated_img + 1) / 2  # Normalize to [0, 1]
            generated_img = torch.clamp(generated_img, 0, 1)  # Clamp values to [0, 1]

            samples.append(generated_img)
            # visualize_denoising_process(sample, path="isfid_gray", filename=f"denoising_process_{i}.png")
            # save_image(generated_img, f'isfid_gray/generated_images_{i + 1}_gray.png')

    return torch.stack(samples)  # This will have shape [num_samples, 1, 32, 32]

def sample_images(num_samples=100, batch_size=10):
    # Calculate how many full batches are needed and any remainder
    num_full_batches = num_samples // batch_size
    remainder = num_samples % batch_size
    
    samples = []
    
    with torch.no_grad():
        # Generate full batches
        for _ in range(num_full_batches):
            sample_batch = diffusion.sample_ddpm(x_shape=(batch_size, 1, 32, 32), device=device)            
            generated_imgs = sample_batch[-1][0]  # Remove the channel dimension if needed
            # Rescale from [-1, 1] to [0, 1] and clamp to ensure valid pixel values
            generated_imgs = (generated_imgs + 1) / 2  # Normalize to [0, 1]
            generated_imgs = torch.clamp(generated_imgs, 0, 1)  # Clamp values to [0, 1]
            samples.append(generated_imgs)
        
        # Generate the remainder if necessary
        if remainder > 0:
            sample_batch = diffusion.sample_ddpm(x_shape=(remainder, 1, 32, 32), device=device)
            generated_imgs = sample_batch[-1][0]
            generated_imgs = (generated_imgs + 1) / 2  # Normalize to [0, 1]
            generated_imgs = torch.clamp(generated_imgs, 0, 1)
            samples.append(generated_imgs)
    
    return torch.cat(samples, dim=0)  # Concatenate all batches into one tensor


# Function to calculate Inception Score (IS)
def inception_score(images, batch_size=32, splits=10):
    n_images = images.size(0)
    scores = []

    with torch.no_grad():
        for i in range(0, n_images, batch_size):
            batch = images[i:i + batch_size].to(device)
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
                pred = classifier(batch)
                activations.append(pred.cpu().numpy())
        return np.concatenate(activations, axis=0)

    # Compute activations for real and generated images
    act_real = get_activations(real_images)
    act_gen = get_activations(generated_images)

    # Calculate means and covariances
    mu_real, sigma_real = np.mean(act_real, axis=0), np.cov(act_real, rowvar=False)
    mu_gen, sigma_gen = np.mean(act_gen, axis=0), np.cov(act_gen, rowvar=False)

    # Calculate FID score
    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid_score
    
       
#Sample generated images
# num_samples = 100
# print(f"Starting to sample {num_samples} images...")
# print(f"Time started: {time.ctime()}")
# start_time = time.time()
# generated_images = sample_images_old(num_samples=num_samples)
# end_time = time.time()
# print(f"Time taken to sample images: {end_time - start_time:.2f} seconds")
# #generated_images = sample_images_wrong(num_samples=num_samples)
# print("Generated images shape:", generated_images.shape)  # Should print [10, 1, 32, 32]

# print("Generated images sampled.")

# Sample generated images
num_samples = 10000  # Total number of images to generate
batch_size = 350 # Number of images generated per batch
print(f"Starting to sample {num_samples} images in batches of {batch_size}...")
print(f"Time started: {time.ctime()}")
start_time = time.time()
generated_images = sample_images(num_samples=num_samples, batch_size=batch_size)
end_time = time.time()
print(f"Time taken to sample images: {end_time - start_time:.2f} seconds")
print("Generated images shape:", generated_images.shape)  # Should print [num_samples, 1, 32, 32]

print("Generated images sampled.")

# Calculate and print IS
is_mean, is_std = inception_score(generated_images)
print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

# Calculate and print FID
fid_score = calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score:.4f}")


for i in range(num_samples):
    if i % 100 == 0:  # Save only 1% of the samples
        save_image(generated_images[i], f'isfid_gray/generated_images_{i + 1}_gray.png')
        print(f"Saved generated image {i + 1}.")