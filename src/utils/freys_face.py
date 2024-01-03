"""
Primarily used to download and loading of the Frey Face dataset.
Perhaps better name "load_freys_face_data"?
Based on a snippet from the web, but made usable by moí.
"""

import os
import pandas as pd
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io


class FreyFaceDataset(Dataset):
    def __init__(self, file_path, transform=None):
        # Load the .mat file
        self.data = scipy.io.loadmat(file_path)["ff"]
        self.transform = transform

    def __len__(self):
        # Return the number of samples
        return self.data.shape[1]

    def __getitem__(self, idx):
        # Get the image data and reshape it appropriately
        image = self.data[:, idx].reshape(
            28, 20
        )  # Adjust the reshape dimensions as necessary
        image = image.astype(
            "float32"
        )  # / 255.0  # Normalize the image to be between 0 and 1

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform specified, we add a channel dimension
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension

        # required for my train loop to work lol
        target = torch.tensor(0)
        return image, target


def fetch_file(url):
    try:
        f = urlopen(url)
        print(f"Downloading data file from {url} to {os.getcwd()}...")

        # Open local file for writing
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())
        print("Download complete!")

    # Handle errors
    except HTTPError as e:
        print("HTTP Error:", e.code, url)
    except URLError as e:
        print("URL Error:", e.reason, url)
    except Exception as e:
        print("Error:", e)


def load_freys_face_data(file_path) -> pd.DataFrame:
    # Reshape data
    img_rows, img_cols = 28, 20
    ff = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    ff = ff["ff"].T.reshape((-1, img_rows * img_cols))
    return pd.DataFrame(ff)


if __name__ == "__main__":
    url = "http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat"
    data_filename = os.path.basename(url)
    if not os.path.exists(data_filename):
        fetch_file(url)

    if os.path.exists(data_filename):
        print(f"Data file {data_filename} found.")
        df = load_freys_face_data(data_filename)
        # Now df contains the loaded data as a DataFrame
        print(df.head())
    else:
        print("Data file could not be downloaded.")
