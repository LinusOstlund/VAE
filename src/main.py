import argparse
from pathlib import Path

import wandb
from tqdm import tqdm
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.VAE import VAE
from utils.losses import evidence_lower_bound
from utils.utils import get_device


def train(model, optimizer, epochs, device, train_loader, loss_function):
    """
    Main training loop
    """
    model.train()
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # print shape of x and the device it is on

            # move x to current device
            x = x.to(device)

            # x shape is [batch_size, 1, 28, 28]
            # flatten x into a vector
            x = x.view(x.size(0), -1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            x_hat, mean, logvar = model(x)

            # backwards pass: compute loss and its gradients
            loss = loss_function(x, x_hat, mean, logvar)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            running_loss += loss.item()

        # at the end of each epoch, log the loss
        epoch_loss = running_loss / len(train_loader.dataset)
        wandb.log({"train_loss": epoch_loss})
        losses.append(epoch_loss)
    return losses


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--batch_size", type=int, default=64)
    argparser.add_argument("--input_dim", type=int, default=784)
    argparser.add_argument("--hidden_dim", type=int, default=400)
    argparser.add_argument("--latent_dim", type=int, default=200)
    argparser.add_argument("--learning_rate", type=float, default=1e-3)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--device", type=str, default=None)
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

    # instantiate model
    model = VAE(
        device=device,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    # setup optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # set up data transformations
    # TODO I am not sure wether we need transformations or not
    transform = transforms.Compose([transforms.ToTensor()])

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

    # evaluera på test set
    # skapa bilder från latent space och spara i wandb, göra några olika traversals? kanske ska ske i inference.py
