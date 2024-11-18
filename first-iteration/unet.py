# U-Net architecture implementation
# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import SinusoidalTimeEmbedding
from utils import visualize_feature_maps
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, embedding_dim=128):
        super(UNet, self).__init__()
        self.time_embedding = SinusoidalTimeEmbedding(embedding_dim)
        self.time_fc1 = nn.Linear(embedding_dim, base_channels)
        self.time_fc2 = nn.Linear(embedding_dim, base_channels * 2)
        self.time_fc3 = nn.Linear(embedding_dim, base_channels * 4)
        self.time_fc4 = nn.Linear(embedding_dim, base_channels * 8)

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        self.dec1 = UpBlock(base_channels * 8, base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2)
        self.dec3 = UpBlock(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self.feature_maps = []

    def forward(self, x, t):
        self.feature_maps.clear()
        # Time embedding
        time_emb = self.time_embedding(t)
        time_emb1 = self.time_fc1(time_emb).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, base_channels, 1, 1]
        time_emb2 = self.time_fc2(time_emb).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, base_channels * 2, 1, 1]
        time_emb3 = self.time_fc3(time_emb).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, base_channels * 4, 1, 1]
        time_emb4 = self.time_fc4(time_emb).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, base_channels * 8, 1, 1]

        # Encoding path
        skip1 = self.enc1(x)
        skip1 = skip1 + time_emb1  # Add the time embedding to the first feature map
        self.feature_maps.append(skip1)

        skip2, x = self.enc2(skip1)
        skip2 = skip2 + time_emb2  # Add the time embedding to the second feature map
        self.feature_maps.append(skip2)

        skip3, x = self.enc3(x)
        skip3 = skip3 + time_emb3  # Add the time embedding to the third feature map
        self.feature_maps.append(skip3)

        # Bottleneck
        x = self.bottleneck(x)
        self.feature_maps.append(x)
        x = x + time_emb4  # Add the time embedding to the bottleneck

        # Decoding path
        x = self.dec1(x, skip3)
        self.feature_maps.append(x)
        x = x + time_emb3  # Add the time embedding after the first up block

        x = self.dec2(x, skip2)
        self.feature_maps.append(x)
        x = x + time_emb2  # Add the time embedding after the second up block

        x = self.dec3(x, skip1)
        self.feature_maps.append(x)
        x = x + time_emb1  # Add the time embedding after the third up block

        return self.final_conv(x)


# Test the U-Net model
if __name__ == "__main__":
    os.makedirs("visualizations/unet", exist_ok=True)
    # Load a single MNIST image with normalization to scale pixel values to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    
    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='./', train=True, transform=transform, download=True)
    
    # Get a single image and label
    image, label = mnist_dataset[0]  # Get the first image in the dataset
    image = image.unsqueeze(0)  # Add a batch dimension (1, 1, 32, 32)

    # Display the original MNIST image (rescaled for visualization)
    plt.imshow((image.squeeze(0).squeeze(0) * 0.5 + 0.5).numpy(), cmap='gray')  # Rescale to [0, 1] for display
    plt.title(f"Original MNIST Image - Label: {label}")
    plt.savefig("visualizations/unet/original_image.png")
    plt.close()

    # Initialize the model and run the image through it
    model = UNet(in_channels=1, out_channels=1, base_channels=64, embedding_dim=128)
    
    # Create a timestep tensor to pass to the model
    timestep = torch.tensor([500])  # Arbitrary timestep value for testing
    output = model(image, timestep)

    # Visualize feature maps captured during the forward pass
    # Uncomment if feature map visualization function exists and is integrated
    visualize_feature_maps(model.feature_maps)

    # Display the output image (rescaled for visualization)
    plt.imshow((output.detach().squeeze(0).squeeze(0) * 0.5 + 0.5).numpy(), cmap='gray')  # Rescale to [0, 1] for display
    plt.title("Output Image from U-Net with Time Embedding")
    plt.savefig("visualizations/unet/output_image.png")
    plt.close()