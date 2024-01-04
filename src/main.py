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
from PIL import Image

from train import train
from models.VAE import VAE
from utils.losses import evidence_lower_bound
from utils.utils import get_device
from utils.utils import convert_grid_to_PIL_image
from utils.freys_face import FreyFaceDataset
from models.encoders_and_decoders import GaussianMLP, BernoulliDecoder


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--batch_size", type=int, default=100)
    argparser.add_argument("--input_dim", type=int, default=784)
    argparser.add_argument("--hidden_dim", type=int, default=500)
    argparser.add_argument("--latent_dim", type=int, default=2)
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        choices=[0.01, 0.02, 0.1],  # taken from the paper
    )
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--device", type=str, default=None)
    argparser.add_argument(
        "--weight_decay", type=float, default=1e-6
    )  # TODO make it cohere to the article
    argparser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "frey"]
    )
    argparser.add_argument(
        "--optimizer", type=str, default="adagrad", choices=["sgd", "adagrad", "adam"]
    )
    args = argparser.parse_args()

    # Convert args to a dictionary
    config_dict = vars(args)

    # Initialize wandb with the config dictionary
    wandb.init(project="DD2434", entity="skolprojekt", config=config_dict)

    # set a tag in wandb
    wandb.run.tags = ["live"]

    # setting this to "none" will default to the "best" available device: CUDA, MPS (Apple M1), or CPU
    device = get_device(device=args.device)

    encoder = GaussianMLP(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.latent_dim
    ).to(device)

    if args.dataset == "mnist":
        decoder = BernoulliDecoder(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.input_dim,
        ).to(device)
    elif args.dataset == "frey":
        decoder = GaussianMLP(
            input_dim=args.latent_dim,
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
    # binarize the data

    # Binarize the data
    transforms = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor first
            transforms.Lambda(lambda x: (x > 0.0).float()),  # Then binarize
        ]
    )

    # download dataset
    data_path = Path("data")

    if args.dataset == "mnist":
        train_dataset = MNIST(
            root=data_path,
            train=True,
            transform=transforms,
            download=True,
        )
        test_dataset = MNIST(data_path, transform=transforms, download=True)
    elif args.dataset == "frey":
        train_dataset = FreyFaceDataset(
            file_path=data_path / Path("frey_rawface.mat"), transform=transforms
        )

    else:
        raise ValueError("Dataset not recognized. Pick 'frey' or 'mnist'.")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # save an image to wandb as a sanity check
    tensor_grid = next(iter(train_loader))[0]
    sanity_check = convert_grid_to_PIL_image(tensor_grid, nrow=4)
    # Log the grid image to wandb
    wandb.log({"Sanity Check": [wandb.Image(sanity_check, caption="Sanity Check")]})

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

    # store the model in wandb
    wandb.save("model.pt")

    # save some plots
    image_height = 28
    image_width = 28
    grid_width = 10
    grid_height = 10

    random_samples = model.sample(
        grid_width=grid_width,
        grid_height=grid_height,
        image_width=image_width,
        image_height=image_height,
    )

    grid_image = convert_grid_to_PIL_image(random_samples, nrow=grid_width)

    # Log the grid image to wandb
    wandb.log({"Random Samples": [wandb.Image(grid_image, caption="Random Samples")]})

    steps = 20
    z_start = 0.01
    z_end = 0.99
    if args.latent_dim == 2:
        # perform a walk in the latent space
        latent_walk = model.latent_walk(
            z_start=z_start,
            z_end=z_end,
            steps=steps,
            image_width=image_width,
            image_height=image_height,
        )
        grid_image = convert_grid_to_PIL_image(latent_walk, nrow=steps)
        # Log the grid image to wandb
        wandb.log({"Latent Walk": [wandb.Image(grid_image, caption="Latent Walk")]})


# evaluera på test set
# skapa bilder från latent space och spara i wandb, göra några olika traversals? kanske ska ske i inference.py
