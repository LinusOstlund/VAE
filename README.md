# VAE

Reproduction of the paper [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) by Kingma and Welling. Uses Weights & Biases to track experiments.

## Usage

Everything is operated from [main.py](main.py). First, you need to install dependencies. I would recommend conda:

```bash
# create a new environment called "vae" with python 3.8
conda create -n vae python=3.8
conda activate vae
pip install -r requirements.txt
```

Then, you need to create a Weights & Biases account and login. You can do this by running:

```bash
wandb login # Should ask for your API key
```

You can also store it in a `.env` file (see [.env.example](.env.example)). A .env file is not included in the repository for security reasons, and must be put in root. I think I will have to add you to my W&B project for it to work, but I'm not sure.

To run the code with defaul parameters, use the following command:

```bash
python src/main.py
```

You can also specify parameters using the following syntax:

```bash
python src/main.py --epochs 100 --batch_size 64
```

To see all available parameters, run:

```bash
python src/main.py --help
```