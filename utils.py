# Utility functions for visualization and other supporting functions
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns

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

def visualize_denoising_process(diffusion_model, steps, x_shape, device='cpu', epoch=None):
    """
    Visualizes the denoising process in one image with subplots for each step.
    
    Args:
        diffusion_model: The diffusion model to use for denoising.
        steps: The number of visualization steps.
        x_shape: The shape of the sample to denoise.
        device: The device to run the denoising on (e.g., 'cpu' or 'cuda').
        epoch: The current epoch number (optional, for labeling purposes).
    """
    x_t = torch.randn(x_shape).to(device)
    os.makedirs("visualizations/denoising_steps", exist_ok=True)

    for t in reversed(range(1, diffusion_model.num_timesteps)):
        t_tensor = torch.full((x_shape[0],), t, dtype=torch.long).to(device)

        noise_pred = diffusion_model.model(x_t, t_tensor)  # Pass the timestep to the model

        beta_t = diffusion_model.beta_scheduler[t].to(device)
        alpha_t = diffusion_model.alpha_scheduler[t].to(device)
        alpha_bar_t = diffusion_model.alpha_bar_scheduler[t].to(device)

        if t > 1:
            noise = torch.randn_like(x_t).to(device)
        else:
            noise = torch.zeros_like(x_t)

        # Update x_t for the next step
        x_t = (
            1 / torch.sqrt(alpha_t)
        ) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise

        # Save intermediate images for inspection
        if t % (diffusion_model.num_timesteps // 10) == 0 or t == 1:  # Save at regular intervals and final step
            plt.imshow((x_t[0].detach().cpu().squeeze() * 0.5 + 0.5).numpy(), cmap='gray')
            plt.title(f"Denoising Step {t}")
            plt.axis('off')
            plt.savefig(f"visualizations/denoising/denoising_{t}.png")
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
