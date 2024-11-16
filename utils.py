# Utility functions for visualization and other supporting functions
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import seaborn as sns
import numpy as np
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
import os
import torch

def visualize_noising_process(image, diffusion_model, steps=10, path="visualizations/noising", filename="noising_process", image_title="Noising Process Visualization"):
    """
    Visualizes the noising process by adding noise at different timesteps.

    Args:
        image (Tensor): The original image (shape: [1, channels, height, width]).
        diffusion_model (DiffusionModel): The initialized diffusion model.
        steps (int): Number of steps to visualize between timestep 0 and the final timestep.
        path (str): Path to save the visualization images.
        filename (str): The filename for the saved visualization.
        image_title (str): Title for the visualization plot.
    """
    # Ensure the image is on the same device as the diffusion model
    device = next(diffusion_model.model.parameters()).device
    image = image.to(device)
    noise = torch.randn_like(image).to(device)
    
    os.makedirs(path, exist_ok=True)
    timesteps = torch.linspace(0, diffusion_model.num_timesteps - 1, steps).long().to(device)

    fig, axes = plt.subplots(1, steps, figsize=(20, 2.5))
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((image.size(0),), t, dtype=torch.long).to(device)
        noisy_image = diffusion_model.add_noise(image, noise, t_tensor)
        
        # Detach and move the image to CPU for visualization
        image_to_display = noisy_image[0].detach().cpu()

        if image_to_display.shape[0] == 3:  # RGB image
            # Rescale and clamp the image to [0, 1] and permute for display
            image_to_display = (image_to_display * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[i].imshow(image_to_display)
        elif image_to_display.shape[0] == 1:  # Grayscale image
            # Rescale and clamp the image to [0, 1]
            image_to_display = (image_to_display.squeeze(0) * 0.5 + 0.5).clamp(0, 1).numpy()
            axes[i].imshow(image_to_display, cmap='gray')
        else:
            raise TypeError(f"Invalid shape {image_to_display.shape} for image data")

        axes[i].axis('off')
        axes[i].set_title(f"t={t.item()}")

    plt.suptitle(image_title)
    plt.tight_layout()
    plt.savefig(f"{path}/{filename}.png")
    plt.close()


def visualize_denoising_process(generated_samples, epoch=None, path="samples/generated_samples", filename="samples_epoch", image_title="Denoising Process Visualization"):
    num_samples = len(generated_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2.5))

    for i, ax in enumerate(axes.flatten()):
        ax.set_title(f"t={generated_samples[i][1]}")
        
        # Detach and move the image to the CPU
        image = generated_samples[i][0].detach().cpu()
        
        # Remove batch dimension if present
        if image.dim() == 4 and image.shape[0] == 1:
            image = image.squeeze(0)  # Remove the batch dimension        

        if image.shape[0] == 3:  # RGB image (CIFAR-10)
            # Rescale and clamp the image to [0, 1] for valid visualization
            image = (image * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(image)
        elif image.shape[0] == 1:  # Grayscale image (MNIST)
            # Rescale and clamp the image to [0, 1]
            image = (image.squeeze(0) * 0.5 + 0.5).clamp(0, 1).numpy()
            ax.imshow(image, cmap='gray')
        else:
            raise TypeError(f"Invalid shape {image.shape} for image data")

        ax.axis('off')
        
    os.makedirs(path, exist_ok=True)
    if epoch is not None:
        filename = f"{filename}_{epoch + 1}"
        image_title = f"{image_title} (Epoch {epoch + 1})"
    plt.suptitle(image_title)
    plt.savefig(f"{path}/{filename}.png")
    plt.close()



def visualize_feature_maps(feature_maps, num_cols=8, visualize_rgb=False):
    """
    Visualizes the feature maps extracted from the U-Net at different layers.

    Args:
        feature_maps (list of Tensors): List of feature maps to visualize.
        num_cols (int): Number of columns for subplot grid.
        visualize_rgb (bool): If True, combine channels to create RGB-like images.
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    os.makedirs("unet", exist_ok=True)

    for i, fmap in enumerate(feature_maps):
        fmap = fmap.squeeze(0)  # Remove the batch dimension
        num_filters = fmap.size(0)  # After squeezing, channels are at dim 0

        if visualize_rgb:
            num_images = num_filters // 3  # Number of RGB images we can create
            num_rows = (num_images // num_cols) + (num_images % num_cols != 0)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))

            # Flatten axes array for easy indexing
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]

            for idx in range(num_images):
                ax = axes[idx]
                # Get the three channels for the RGB image
                rgb_channels = fmap[idx * 3:(idx + 1) * 3].detach().cpu().numpy()

                # Ensure we have exactly 3 channels
                if rgb_channels.shape[0] < 3:
                    # Not enough channels to form an RGB image
                    break

                # Normalize the RGB channels
                rgb_image = (rgb_channels - rgb_channels.min()) / (rgb_channels.max() - rgb_channels.min() + 1e-5)
                rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Shape: H x W x 3

                ax.imshow(rgb_image)
                ax.axis('off')

            # Hide any unused subplots
            for ax in axes[num_images:]:
                ax.axis('off')

        else:
            # Visualize each feature map individually in grayscale
            num_rows = (num_filters // num_cols) + (num_filters % num_cols != 0)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))

            # Flatten axes array for easy indexing
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]

            for idx in range(num_filters):
                ax = axes[idx]
                feature_map = fmap[idx].detach().cpu().numpy()
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-5)
                ax.imshow(feature_map, cmap='gray')
                ax.axis('off')

            # Hide any unused subplots
            for ax in axes[num_filters:]:
                ax.axis('off')

        plt.suptitle(f"Feature Maps at Layer {i + 1}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"unet/feature_maps_{i + 1}.png")
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

