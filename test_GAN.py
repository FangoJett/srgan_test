import os
import math
import argparse
from os import listdir
from os.path import join
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose, Normalize
from PIL import Image
import matplotlib.pyplot as plt

# Sprawdzenie, czy plik jest obrazem
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

# Definicje klas modelu
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        return (torch.tanh(block8) + 1) / 2


# Dataset
class FullImageDataset(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(FullImageDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.lr_transform = Compose([
            ToPILImage(),
            lambda img: Resize((img.height // upscale_factor, img.width // upscale_factor), interpolation=Image.BILINEAR)(img),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.hr_transform = Compose([ToTensor()])

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        hr_image = self.hr_transform(hr_image)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


# Denormalizacja
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# Wizualizacja obrazów
def visualize_images(lr_image, generated_hr, true_hr=None):
    to_pil = ToPILImage()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    lr_image = denormalize(lr_image.clone(), mean, std)
    lr_image = torch.clamp(lr_image, 0, 1)
    fig, axes = plt.subplots(1, 3 if true_hr is not None else 2, figsize=(15, 5))

    axes[0].imshow(to_pil(lr_image), cmap='gray')
    axes[0].set_title("Low-Resolution (LR)")
    axes[0].axis("off")

    axes[1].imshow(to_pil(generated_hr), cmap='gray')
    axes[1].set_title("Generated High-Resolution (SR)")
    axes[1].axis("off")

    if true_hr is not None:
        axes[2].imshow(to_pil(true_hr), cmap='gray')
        axes[2].set_title("True High-Resolution (HR)")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# Główna funkcja
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-Resolution Image Generator")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Folder with test images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained generator model")
    parser.add_argument("--upscale_factor", type=int, default=4, help="Upscaling factor for super-resolution")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (cuda or cpu)")
    args = parser.parse_args()

    #Przygotowanie modelu
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    netG = torch.load(args.model_path, map_location=device)
    netG.eval()

    # Przygotowanie datasetu
    test_dataset = FullImageDataset(args.dataset_dir, args.upscale_factor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Testowanie i wizualizacja
    for lr_image, true_hr in test_loader:
        lr_image = lr_image.to(device)
        true_hr = true_hr.to(device)

        with torch.no_grad():
            generated_hr = netG(lr_image)

        visualize_images(lr_image.cpu().squeeze(0), generated_hr.cpu().squeeze(0), true_hr.cpu().squeeze(0))
