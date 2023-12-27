import argparse
import numpy as np
from pathlib import Path

import wandb
from tqdm import tqdm
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image

from models.VAE import VAE
from utils.losses import evidence_lower_bound
from utils.utils import get_device
from models.encoders_and_decoders import GaussianMLP, BernoulliDecoder


def show_images(images):
    """
    Takes a batch of images from a train_loader, deatches them and converts them to numpy arrays. Then it plots them on a little grid.
    TODO flytta till en util
    """
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    import matplotlib.pyplot as plt

    images = images.detach().cpu().numpy()
    images = np.reshape(images, (-1, 28, 28))
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0.1)
    for ax, im in zip(grid, images):
        ax.imshow(im, cmap="gray")
    # save image as "sanity_check.png"
    plt.savefig("plots/sanity_check.png")


def train(model, optimizer, epochs, device, train_loader, loss_function):
    """
    Main training loop
    """
    model.train()
    wandb.watch(model, log="all")
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # x shape is [batch_size, channels, 28, 28]
            batch_size, _, width, height = x.shape

            # vectorize image
            x = x.reshape(batch_size, width * height).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            x_hat, mean, logvar = model(x)
            # assert that the mean and logvar contains no NaNs
            assert not torch.isnan(mean).any()

            # backwards pass: compute loss and its gradients
            reconstruction_loss, DKL = loss_function(x, x_hat, mean, logvar)

            elbo = reconstruction_loss + DKL
            elbo.backward()

            # Adjust learning weights
            optimizer.step()

            running_loss += elbo.item()
            # logg my three losses
            wandb.log(
                {
                    "train": {
                        "reconstruction_loss": reconstruction_loss.item(),
                        "DKL": DKL.item(),
                        "elbo": elbo.item(),
                    }
                },
            )

        # at the end of each epoch, log the loss
        epoch_loss = running_loss / len(train_loader.dataset)
        wandb.log({"train": {"loss": epoch_loss}}, step=epoch)
        losses.append(epoch_loss)
    return losses


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--input_dim", type=int, default=784)
    argparser.add_argument("--hidden_dim", type=int, default=400)
    argparser.add_argument("--latent_dim", type=int, default=2)
    argparser.add_argument("--learning_rate", type=float, default=1e-4)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--device", type=str, default=None)
    argparser.add_argument(
        "--weight_decay", type=float, default=1e-6
    )  # TODO make it cohere to the article
    argparser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "frey"]
    )
    argparser.add_argument(
        "--optimizer", type=str, default="sgd", choices=["sgd", "adagrad", "adam"]
    )
    args = argparser.parse_args()

    wandb.init(
        project="DD2434",
        entity="skolprojekt",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "seed": args.seed,
            "device": args.device,
            "dataset": args.dataset,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
        },
    )

    # setting this to "none" will default to the "best" available device: CUDA, MPS (Apple M1), or CPU
    device = get_device(device=args.device)

    encoder = GaussianMLP(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.latent_dim
    ).to(device)

    decoder = BernoulliDecoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.input_dim,
    ).to(device)

    # instantiate model
    model = VAE(
        device=device,
        encoder=encoder,
        decoder=decoder,
    ).to(device)

    # setup optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

    # set up data transformations
    # TODO vi använder MNIST som redan är normaliserat mellan [0,1], vi vill egentligen ha [-1,1] för att matcha artikeln... hur fixa. Eller vil lvi ens det? :D
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # download dataset
    data_path = ".." / Path("data")

    if args.dataset == "mnist":
        train_dataset = MNIST(
            root=data_path, train=True, transform=transforms.ToTensor(), download=True
        )
        test_dataset = MNIST(data_path, transform=transform, download=True)
    elif args.dataset == "frey":
        raise NotImplementedError("Frey dataset not implemented yet.")
    else:
        raise ValueError("Dataset not recognized. Pick 'frey' or 'mnist'.")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # setup loss (higher order function)
    elbo = evidence_lower_bound

    losses = train(
        model=model,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        train_loader=train_loader,
        loss_function=elbo,
    )

    # store the model in wandb (not only state_dict)
    wandb.save("model.pt")

    # TODO nedan är copy pastad kod.. metodifiera
    steps = 10
    z_start = -2.0
    z_end = 2.0

    latent_walk = model.latent_walk(z_start=z_start, z_end=z_end, steps=steps)

    width = 10
    height = 10
    random_samples = model.sample(width=width, height=height)

    # Create a grid of images
    grid = make_grid(random_samples, nrow=height).cpu().detach()

    # Convert the grid to a PIL Image
    grid_np = grid.numpy().transpose(
        (1, 2, 0)
    )  # Convert from PyTorch format (C, H, W) to numpy format (H, W, C)
    grid_image = Image.fromarray(
        np.uint8(grid_np * 255)
    )  # Scale to 0-255 and convert to uint8

    # Log the grid image to wandb
    wandb.log({"Random Samples": [wandb.Image(grid_image, caption="Random Samples")]})

    # Create a grid of images
    grid = make_grid(latent_walk, nrow=steps).cpu().detach()

    # Convert the grid to a PIL Image
    grid_np = grid.numpy().transpose(
        (1, 2, 0)
    )  # Convert from PyTorch format (C, H, W) to numpy format (H, W, C)
    grid_image = Image.fromarray(
        np.uint8(grid_np * 255)
    )  # Scale to 0-255 and convert to uint8

    # Log the grid image to wandb
    wandb.log({"Latent Walk": [wandb.Image(grid_image, caption="Latent Walk")]})


# evaluera på test set
# skapa bilder från latent space och spara i wandb, göra några olika traversals? kanske ska ske i inference.py
