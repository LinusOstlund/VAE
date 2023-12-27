import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMLP(nn.Module):
    """
    MLP with Gaussian output
    Can be used as decoder or encoder
    Used as a decoder for FreyFace
    Correpsonding to Appendix C.1 in the paper
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GaussianMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, output_dim)
        self.fc2_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        """
        Input can be either an image (decoder) or a latent vector (encoder)
        """
        h = torch.tanh(self.fc1(input))
        mean = self.fc2_mean(h)
        log_var = self.fc2_var(h)
        return mean, log_var


class BernoulliDecoder(nn.Module):
    """
    Bernoulli MLP decoder
    Used for MNIST
    Corresponding to Appendix C.2 in the paper
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(BernoulliDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.tanh(self.fc1(z))
        y = torch.sigmoid(self.fc2(h))
        return y
