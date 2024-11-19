# metrics2.py
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms, datasets  # Ensure datasets is imported
from PIL import Image
from classifier import SimpleCNN
from scipy.linalg import sqrtm
import os

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom dataset for loading images from a directory
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale if needed
        if self.transform:
            image = self.transform(image)
        return image

# Transform for pre-processed images
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure size compatibility
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load real and generated image datasets
real_dataset = datasets.MNIST(root='./', train=False, transform=transform, download=True)
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
real_images = torch.cat([batch[0] for batch in real_loader])

generated_dataset = ImageDataset(image_dir='isfid_gray', transform=transform)
generated_loader = DataLoader(generated_dataset, batch_size=32, shuffle=False)

# Collect images and ensure all have the same shape and channels
generated_images = []

for img_batch in generated_loader:
    for img in img_batch:
        # Ensure the image is grayscale (1 channel)
        if img.shape[0] != 1:
            print(f"Image with incorrect number of channels found: {img.shape}")
            img = img.mean(dim=0, keepdim=True)  # Convert to single channel if needed
        generated_images.append(img.unsqueeze(0))

# Concatenate images to form a single tensor
generated_images = torch.cat(generated_images, dim=0)
print(f"Loaded {len(real_images)} real images.")
print(f"Loaded {len(generated_images)} generated images.")

# Load the pre-trained SimpleCNN model
classifier = SimpleCNN().to(device)
classifier.load_state_dict(torch.load("mnist_classifier.pth", weights_only=True))
classifier.eval()

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

# Calculate and print IS
is_mean, is_std = inception_score(generated_images)
print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

# Calculate and print FID
fid_score = calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score:.4f}")
