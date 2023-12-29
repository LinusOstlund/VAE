import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
    ):
        super(VAE, self).__init__()
        self.device = device
        # The Encoder
        # q(z|x), weights: phi
        self.encoder = encoder

        # The Decoder (theta)
        # p(x|z), weights: theta
        self.decoder = decoder

    def encode(self, x):
        mean, log_var = self.encoder(x)
        return mean, log_var

    def reparameterization(self, mean, var):
        std = torch.exp(0.5 * var)
        epsilon = torch.randn_like(std).to(self.device)
        z = mean + epsilon * std
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        # ifall decodern är GaussianMLP så blir x_hat = mean & logvar
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def sample(self, width: int, height: int, **kwargs) -> torch.Tensor:
        """
        Sample from the prior distribution (standard Gaussian) and generate new data. Width and height of the resulting grid of images (in number of images) can be specified.

        Example usage:
        >>> batch = model.sample(width=7, height=7)
        >>> grid = make_grid(batch, nrow=7)
        >>> show(grid)
        """
        latent_dim = self.encoder.fc2_mean.out_features
        z = torch.randn(width * height, latent_dim).to(self.device)
        generated_data = self.decode(z)
        return generated_data.reshape(width * height, 1, 28, 28)

    def latent_walk(self, z_start, z_end, steps):
        """
        Perform a latent walk in the latent space. Use with 2D latent space, and Pytorch visions make_grid function to visualize the walk.

        Example usage:
        >>> steps = 10
        >>> generated_data = model.latent_walk(-2, 2, steps=steps)
        >>> grid = make_grid(generated_data, nrow=steps)
        >>> show(grid) # show from utils/utils.py

        Args:
        - z_start (int): Starting point in latent space.
        - z_end (int): Ending point in latent space.
        - steps (int): Number of steps in the walk.

        Returns:
        - torch.Tensor: A tensor containing the generated data at each step of the latent walk, shape (steps**2, 1, 28, 28)
        """

        # Linear interpolation between the start and end points to make a grid of xs, ys
        z_grid = torch.stack(
            [
                torch.stack([x, y], dim=0)
                for x in torch.linspace(z_start, z_end, steps).to(self.device)
                for y in torch.linspace(z_start, z_end, steps).to(self.device)
            ],
            dim=0,
        )

        # Decode each point along the walk
        generated_data = torch.stack([self.decode(z) for z in z_grid], dim=0)

        return generated_data.reshape(steps**2, 1, 28, 28)
