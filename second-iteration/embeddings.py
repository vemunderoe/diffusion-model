# Sinusoidal time embedding
import torch
import torch.nn as nn
from utils import visualize_time_embeddings

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        device = timesteps.device  # Ensure emb_scale is on the same device as timesteps
        half_dim = self.embedding_dim // 2
        emb_scale = torch.exp(-torch.arange(0, half_dim, dtype=torch.float32, device=device) * 
                              (torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)))
        emb = timesteps[:, None] * emb_scale[None, :]
        
        # Concatenate sin and cos embeddings
        time_embedding = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return time_embedding

if __name__ == "__main__":    
    # Example usage to visualize time embeddings
    timesteps = torch.arange(0, 50).float()  # Example timesteps from 0 to 49
    embedding_dim = 128  # Example embedding dimension

    # Instantiate the SinusoidalTimeEmbedding class
    sinusoidal_time_embedding = SinusoidalTimeEmbedding(embedding_dim)

    # Pass the timesteps tensor through the class instance to generate embeddings
    time_embeddings = sinusoidal_time_embedding(timesteps)

    # Visualize the generated time embeddings
    visualize_time_embeddings(timesteps, embedding_dim, lambda t, d: sinusoidal_time_embedding(t))
