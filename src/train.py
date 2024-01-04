import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm


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

            elbo = reconstruction_loss - DKL

            elbo.backward()
            optimizer.step()
            running_loss += elbo.item()
            # logg my three losses
            wandb.log(
                {
                    "train": {
                        "reconstruction_loss": reconstruction_loss.item(),
                        "DKL": DKL.item(),
                        # logging the negative elbo...
                        "elbo": -elbo.item(),
                    }
                },
            )

        # at the end of each epoch, log the loss
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
    return losses
