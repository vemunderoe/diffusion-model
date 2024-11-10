# Implementation of ELBO loss
import torch
import torch.nn as nn

class ELBOLoss(nn.Module):
    """
    Implements the ELBO loss for diffusion models, specifically for denoising autoencoders.
    """
    def __init__(self, beta_scheduler):
        """
        Initializes the ELBO loss class.

        Args:
            beta_scheduler (Tensor): A tensor representing the beta schedule for the diffusion process.
        """
        super(ELBOLoss, self).__init__()
        self.beta_scheduler = beta_scheduler  # A tensor of shape [num_timesteps]
        self.mse_loss = nn.MSELoss(reduction='none')  # No reduction for element-wise loss

    def forward(self, predicted, target, timestep):
        """
        Computes the ELBO loss for a given predicted and target noise at a specific timestep.

        Args:
            predicted (Tensor): The predicted noise from the U-Net (e.g., [batch_size, channels, height, width]).
            target (Tensor): The actual noise added to the data (e.g., [batch_size, channels, height, width]).
            timestep (Tensor): The current timestep (e.g., [batch_size]).

        Returns:
            Tensor: The computed ELBO loss for the current batch.
        """
        # Calculate the MSE between predicted and true noise
        mse_loss = self.mse_loss(predicted, target)  # Shape: [batch_size, channels, height, width]

        # Get the current beta value based on the timestep
        beta_t = self.beta_scheduler[timestep].view(-1, 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]

        # Weight the MSE loss by beta and sum over all dimensions except batch
        weighted_loss = beta_t * mse_loss
        loss = weighted_loss.mean()  # Average over the batch

        return loss

# Example usage in training or testing
if __name__ == "__main__":
    # Example beta scheduler: linear schedule from 0.0001 to 0.02 over 1000 timesteps
    beta_scheduler = torch.linspace(0.0001, 0.02, 1000)

    # Generate random predicted and true noise tensors for testing
    predicted_noise = torch.randn(4, 1, 32, 32)  # Example shape [batch_size, channels, height, width]
    true_noise = torch.randn(4, 1, 32, 32)       # Same shape as predicted noise
    timestep = torch.randint(0, 1000, (4,))      # Random timesteps for each sample in the batch

    # Initialize the ELBO loss function
    loss_fn = ELBOLoss(beta_scheduler)

    # Compute the loss
    loss = loss_fn(predicted_noise, true_noise, timestep)

    # Print the computed loss
    print(f"Computed ELBO Loss: {loss.item()}")
