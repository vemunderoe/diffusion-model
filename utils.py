# Utility functions for visualization and other supporting functions
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import seaborn as sns
import numpy as np
from scipy.linalg import sqrtm

def visualize_noising_process(image, diffusion_model, steps=10):
    """
    Visualizes the noising process by adding noise at different timesteps.

    Args:
        image (Tensor): The original image (shape: [1, channels, height, width]).
        diffusion_model (DiffusionModel): The initialized diffusion model.
        steps (int): Number of steps to visualize between timestep 0 and the final timestep.
    """
    os.makedirs("visualizations/noising", exist_ok=True)
    timesteps = torch.linspace(0, diffusion_model.num_timesteps - 1, steps).long()
    noise = torch.randn_like(image)

    fig, axes = plt.subplots(1, steps, figsize=(15, 5))
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((image.size(0),), t, dtype=torch.long)
        noisy_image = diffusion_model.add_noise(image, noise, t_tensor)
        axes[i].imshow((noisy_image[0].squeeze(0).detach().cpu().numpy() * 0.5 + 0.5), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"t={t.item()}")

    plt.suptitle("Noising Process Visualization")
    plt.tight_layout()
    plt.savefig("visualizations/noising/noising_process.png")
    plt.close()

def visualize_denoising_process(generated_samples, epoch):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(f"t={generated_samples[i][1]}")
        ax.imshow((generated_samples[i][0].detach().cpu().squeeze() * 0.5 + 0.5).numpy(), cmap='gray')
        ax.axis('off')
            
    plt.suptitle(f"Generated Samples - Epoch {epoch + 1}")
    os.makedirs("samples/generated_samples", exist_ok=True)
    plt.savefig(f"samples/generated_samples/samples_epoch_{epoch + 1}.png")
    plt.close()


def visualize_feature_maps(feature_maps, num_cols=8):
    os.makedirs("visualizations/unet", exist_ok=True)
    for i, fmap in enumerate(feature_maps):
        num_filters = fmap.size(1)
        fmap = fmap.squeeze(0)  # Remove the batch dimension

        num_rows = (num_filters // num_cols) + (num_filters % num_cols != 0)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))
        
        for j in range(num_filters):
            ax = axes[j // num_cols, j % num_cols]
            ax.imshow(fmap[j].detach().cpu().numpy(), cmap='gray')
            ax.axis('off')

        for ax in axes.flat[num_filters:]:
            ax.axis('off')

        plt.suptitle(f"Feature Maps at Layer {i + 1}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"visualizations/unet/feature_maps_{i + 1}.png")
        plt.close()

def visualize_time_embeddings(timesteps, embedding_dim, embedding_function):
    embeddings = embedding_function(timesteps, embedding_dim)
    embeddings_np = embeddings.detach().cpu().numpy()

    plt.figure(figsize=(15, 6))
    sns.heatmap(embeddings_np, cmap='coolwarm', cbar=True, xticklabels=10)
    plt.title(f'Sinusoidal Time Embeddings Heatmap (Timesteps x Embedding Depth)')
    plt.xlabel('Depth')
    plt.ylabel('Position')
    plt.savefig("visualizations/time_embeddings.png")
    plt.close()

def inception_score(images, classifier, num_splits=10):
    """
    Compute the Inception Score (IS) for generated images.
    Args:
        images (Tensor): Generated images, shape [num_images, channels, height, width].
        classifier (nn.Module): Pre-trained classifier that outputs logits or probabilities.
        num_splits (int): Number of splits for score calculation.
    Returns:
        mean (float): Mean of IS scores.
        std (float): Standard deviation of IS scores.
    """
    # Pass images through classifier and get softmax outputs
    with torch.no_grad():
        preds = F.softmax(classifier(images), dim=1)
    
    split_scores = []
    for k in range(num_splits):
        part = preds[k * (len(images) // num_splits): (k + 1) * (len(images) // num_splits), :]
        kl = part * (torch.log(part) - torch.log(torch.mean(part, dim=0, keepdim=True)))
        kl = kl.sum(dim=1).mean()
        split_scores.append(kl.exp())

    return torch.mean(torch.stack(split_scores)).item(), torch.std(torch.stack(split_scores)).item()

def calculate_fid(real_features, fake_features):
    """
    Compute the Fréchet Inception Distance (FID) between real and generated images.
    Args:
        real_features (np.array): Feature representations from real images, shape [num_images, feature_dim].
        fake_features (np.array): Feature representations from generated images, shape [num_images, feature_dim].
    Returns:
        fid (float): The computed FID score.
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

