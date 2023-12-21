import torch
import torch.nn as nn


def evidence_lower_bound(x, x_hat, mean, log_var):
    """
    ELBO for Bernoulli VAE
    """
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    DKL = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reconstruction_loss + DKL
