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
