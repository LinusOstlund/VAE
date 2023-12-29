import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision.utils import make_grid


def get_device(device) -> torch.device:
    """
    Returns the best available device if unspecified: CUDA, MPS, or CPU
    """
    # Check if CUDA is available
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")

        # Check for Apple's Metal Performance Shaders (MPS) for M1/M2 Macs
        elif "mps" in torch.backends.__dict__ and torch.backends.mps.is_available():
            return torch.device("mps")

        else:
            return torch.device("cpu")

    # Default to CPU
    else:
        return torch.device(device)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def convert_grid_to_PIL_image(tensor_grid: torch.Tensor, nrow: int) -> Image.Image:
    """
    Convert a grid of vectorized images to a PIL image.
    Used for storing images in wandb.
    Args:
        tensor_grid: grid of vectorized images.
        nrow: number of rows in the grid.
    """

    # Unnormalize the images (reverse the normalization)
    # tensor_grid = tensor_grid * 0.1307 + 0.3081

    # Create a grid of images
    grid = make_grid(tensor_grid, nrow=nrow).cpu().detach()

    # Convert the grid to a PIL Image
    # PyTorch format (C, H, W) to NumPy format (H, W, C)
    grid_np = grid.numpy().transpose((1, 2, 0))

    grid_image = Image.fromarray(
        np.uint8(grid_np * 255)
    )  # Scale to 0-255 and convert to uint8

    return grid_image
