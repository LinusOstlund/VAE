import torch
import torch.nn as nn
import torch.nn.functional as F


def evidence_lower_bound(x, x_hat, mean, log_var):
    """
    ELBO for Bernoulli VAE
    """
    reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum")

    # Klippa gradienten här, kanske?
    # log var
    DKL = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reconstruction_loss, DKL
