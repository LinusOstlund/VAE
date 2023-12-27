import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


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
