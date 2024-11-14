# metrics.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from torchvision import transforms, datasets
from classifier import SimpleCNN
from diffusion import DiffusionModel
from unet import UNet
from beta_scheduler import BetaScheduler
from scipy.linalg import sqrtm

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pre-trained SimpleCNN model
classifier = SimpleCNN().to(device)
classifier.load_state_dict(torch.load("mnist_classifier.pth", weights_only=True))
classifier.eval()

# Load the trained U-Net model
model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128).to(device)
model.load_state_dict(torch.load("checkpoints/unet_diffusion_model.pth", map_location=device, weights_only=True))
model.eval()

# Define beta scheduler and diffusion model parameters
num_timesteps = 1000
scheduler = BetaScheduler(num_timesteps, device=device)
beta_scheduler = scheduler.get_schedule("linear")
diffusion = DiffusionModel(model, beta_scheduler, num_timesteps=num_timesteps)

def sample_images(num_samples=100):
    samples = []
    for i in range(num_samples):
        with torch.no_grad():
            sample = diffusion.sample_ddpm(x_shape=(1, 1, 32, 32), device=device)
            samples.append(sample[-1][0].squeeze(0))  # Remove the extra dimension

    return torch.stack(samples)  # This will have shape [num_samples, 1, 32, 32]


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
    

# Example usage
if __name__ == "__main__":          
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
    # Sample generated images
    #generated_images = sample_images(num_samples=len(real_images))
    generated_images = sample_images(num_samples=1000)
    print("Generated images shape:", generated_images.shape)  # Should print [10, 1, 32, 32]


    print("Generated images sampled.")

    # Calculate and print IS
    is_mean, is_std = inception_score(generated_images)
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

    # Calculate and print FID
    fid_score = calculate_fid(real_images, generated_images)
    print(f"FID Score: {fid_score:.4f}")
