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
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        return F.relu(x + self.conv_block(x))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma * out + x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention = SelfAttention(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        p = self.pool(x)
        return x, p

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.attention = SelfAttention(out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, embedding_dim=128, num_classes=None):
        super(UNet, self).__init__()
        self.time_embedding = SinusoidalTimeEmbedding(embedding_dim)
        self.time_fc1 = nn.Linear(embedding_dim, base_channels)
        self.time_fc2 = nn.Linear(embedding_dim, base_channels * 2)
        self.time_fc3 = nn.Linear(embedding_dim, base_channels * 4)
        self.time_fc4 = nn.Linear(embedding_dim, base_channels * 8)

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = nn.Sequential(
            DoubleConv(base_channels * 4, base_channels * 8),
            SelfAttention(base_channels * 8)
        )

        self.dec1 = UpBlock(base_channels * 8, base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2)
        self.dec3 = UpBlock(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.feature_maps = []

        if num_classes is not None:
            self.label_embedding = nn.Embedding(num_classes, embedding_dim)
            self.label_emdedding_fc1 = nn.Linear(embedding_dim, base_channels)
            self.label_emdedding_fc2 = nn.Linear(embedding_dim, base_channels * 2)
            self.label_emdedding_fc3 = nn.Linear(embedding_dim, base_channels * 4)
            self.label_emdedding_fc4 = nn.Linear(embedding_dim, base_channels * 8)

    def forward(self, x, t, y=None):
        self.feature_maps.clear()
        # Time embedding
        time_emb = self.time_embedding(t)
        time_emb1 = self.time_fc1(time_emb).unsqueeze(-1).unsqueeze(-1)
        time_emb2 = self.time_fc2(time_emb).unsqueeze(-1).unsqueeze(-1)
        time_emb3 = self.time_fc3(time_emb).unsqueeze(-1).unsqueeze(-1)
        time_emb4 = self.time_fc4(time_emb).unsqueeze(-1).unsqueeze(-1)

        if y is not None:
            label_emb = self.label_embedding(y)
            label_emb = label_emb.view(label_emb.size(0), -1)  # Ensure the label embedding is flat
            label_emb1 = self.label_emdedding_fc1(label_emb).unsqueeze(-1).unsqueeze(-1)
            time_emb1 = time_emb1 + label_emb1
            label_emb2 = self.label_emdedding_fc2(label_emb).unsqueeze(-1).unsqueeze(-1)
            time_emb2 = time_emb2 + label_emb2
            label_emb3 = self.label_emdedding_fc3(label_emb).unsqueeze(-1).unsqueeze(-1)
            time_emb3 = time_emb3 + label_emb3
            label_emb4 = self.label_emdedding_fc4(label_emb).unsqueeze(-1).unsqueeze(-1)
            time_emb4 = time_emb4 + label_emb4

        # Encoding path
        skip1 = self.enc1(x)
        skip1 = skip1 + time_emb1
        self.feature_maps.append(skip1)

        skip2, x = self.enc2(skip1)
        skip2 = skip2 + time_emb2
        self.feature_maps.append(skip2)

        skip3, x = self.enc3(x)
        skip3 = skip3 + time_emb3
        self.feature_maps.append(skip3)

        # Bottleneck
        x = self.bottleneck(x)
        x = x + time_emb4
        self.feature_maps.append(x)

        # Decoding path
        x = self.dec1(x, skip3)
        x = x + time_emb3
        self.feature_maps.append(x)

        x = self.dec2(x, skip2)
        x = x + time_emb2
        self.feature_maps.append(x)

        x = self.dec3(x, skip1)
        x = x + time_emb1
        self.feature_maps.append(x)

        return self.final_conv(x)


# Test the U-Net model
if __name__ == "__main__":
    os.makedirs("unet", exist_ok=True)
    
    # Load a single CIFAR-10 image with normalization to scale pixel values to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 images are already 32x32, so this is optional
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels to range [-1, 1]
    ])
    
    # Load the CIFAR-10 dataset
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    
    # Get a single image and label
    image, label = cifar10_dataset[0]  # Get the first image in the dataset
    image = image.unsqueeze(0)  # Add a batch dimension (1, 3, 32, 32)

    # Display the original CIFAR-10 image (rescaled for visualization)
    plt.imshow((image.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5).numpy())  # Rescale to [0, 1] for display and permute for RGB display
    plt.title(f"Original CIFAR-10 Image - Label: {label}")
    plt.savefig("unet/original_image.png")
    plt.close()

    # Initialize the model and run the image through it
    model = UNet(in_channels=3, out_channels=3, base_channels=32, embedding_dim=64)
    
    # Create a timestep tensor to pass to the model
    timestep = torch.tensor([500])  # Arbitrary timestep value for testing
    output = model(image, timestep)

    # Visualize feature maps captured during the forward pass (if integrated)
    # Uncomment if feature map visualization function exists and is integrated
    visualize_feature_maps(model.feature_maps, visualize_rgb=True)

    # Display the output image (rescaled for visualization)
    plt.imshow((output.detach().squeeze(0).permute(1, 2, 0) * 0.5 + 0.5).numpy())  # Rescale to [0, 1] for display
    plt.title("Output Image from U-Net with Time Embedding")
    plt.savefig("unet/output_image.png")
    plt.close()


