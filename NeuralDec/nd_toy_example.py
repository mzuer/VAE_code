# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/kasparmartens/NeuralDecomposition/blob/master/toy_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + id="-K4LKsQXDiyb" colab_type="code" colab={}
# !pip install --upgrade git+https://github.com/kasparmartens/NeuralDecomposition.git

# + [markdown] id="_uWk9bnGZw8r" colab_type="text"
# Necessary imports

# + id="19UUaL9pUkE9" colab_type="code" colab={}
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from ND.encoder import cEncoder
from ND.decoder import Decoder
from ND.CVAE import CVAE
from ND.helpers import expand_grid

from torch.utils.data import TensorDataset, DataLoader

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

# + [markdown] id="FrNDWlC7Z8TV" colab_type="text"
# Choose device (i.e. CPU or GPU)

# + id="RBWe-V94Zvwp" colab_type="code" colab={}
device = "cuda"

# + [markdown] id="7YcLw4FuZ-vF" colab_type="text"
# Generate a synthetic data set (700 data points, 5 features)

# + id="vxcIxlcPiuPW" colab_type="code" colab={}
N = 700

# generate ground truth latent variable z and covariate c
z = Uniform(-2.0, 2.0).sample((N, 1))
c = Uniform(-2.0, 2.0).sample((N, 1))
noise_sd = 0.05

# generate five features
y1 = torch.exp(-z**2) - 0.2*c
y2 = torch.sin(z) + 0.2*c + 0.7*torch.sin(z)*(z > 0).float()*c
y3 = torch.tanh(z) + 0.2*c
y4 = 0.2*z + torch.tanh(c)
y5 = 0.1*z

Y = torch.cat([y1, y2, y3, y4, y5], dim=1)
Y += noise_sd * torch.randn_like(Y)

Y = (Y - Y.mean(axis=0, keepdim=True)) / Y.std(axis=0, keepdim=True)

data_dim = Y.shape[1]
n_covariates = 1
hidden_dim = 32

# + [markdown] id="4rNqYQ5IaLDD" colab_type="text"
# For model fitting, we will need a `DataLoader` object

# + id="HdLDDWgEvTMW" colab_type="code" colab={}
dataset = TensorDataset(Y.to(device), c.to(device))
data_loader = DataLoader(dataset, shuffle=True, batch_size=64)

# + [markdown] id="BnUVQA7jZSpP" colab_type="text"
# Setting up the CVAE encoder + decoder

# + id="pJSz9pROyAO_" colab_type="code" colab={}
### ENCODER

# define encoder which maps (data, covariate) -> (z_mu, z_sigma)
encoder_mapping = nn.Sequential(
    nn.Linear(data_dim + n_covariates, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 2)
)

encoder = cEncoder(z_dim=1, mapping=encoder_mapping)

# + id="FUesg8PHQWxn" colab_type="code" colab={}
### DECOMPOSABLE DECODER

# grid needed for quadrature
grid_z = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)
grid_c = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)
grid_cz = torch.cat(expand_grid(grid_z, grid_c), dim=1).to(device)

decoder_z = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder_c = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder_cz = nn.Sequential(
    nn.Linear(2, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder = Decoder(data_dim, 
                  grid_z, grid_c, grid_cz, 
                  decoder_z, decoder_c, decoder_cz,
                  has_feature_level_sparsity=True, p1=0.1, p2=0.1, p3=0.1, 
                  lambda0=1e2, penalty_type="MDMM",
                  device=device)

# + [markdown] id="DB9uqHSVZgT_" colab_type="text"
# Combine the encoder + decoder and fit the decomposable CVAE

# + id="T-oOa_T2QYKd" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 907} outputId="02f18402-68d7-4685-e524-e2d2a8eeb23d"
model = CVAE(encoder, decoder, lr=5e-3, device=device)

loss, integrals = model.optimize(data_loader,
                                 n_iter=25000, 
                                 augmented_lagrangian_lr=0.1)


# + [markdown] id="m_4crls9Hou0" colab_type="text"
# ### Diagnostics and interpretation of the model fit

# + [markdown] id="glOKZ-QZHvz3" colab_type="text"
# First let's see if the integrals have converged sufficiently close to zero

# + id="j4NqPJgBs1Xn" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="8c2b57a3-9c16-43f1-8d63-4dd29485e546"
def plot_integrals(integrals):
    n_rep = integrals.shape[0]
    n_iter = integrals.shape[1]
    time = np.arange(n_iter).reshape(-1, 1)
    time_mat = np.tile(time, [1, n_rep])

    plt.plot(time_mat, integrals.T, c="black", alpha=0.25)
    plt.ylim(-0.5, 0.5)

plot_integrals(integrals)

# + [markdown] id="Hk9C7wCHH7PJ" colab_type="text"
# Now let's look at the inferred $z$ values, together with the mappings $z \mapsto \text{features}$

# + id="UEVeue2dzJfZ" colab_type="code" colab={}
with torch.no_grad():
    # encoding of the entire observed data set
    mu_z, sigma_z = encoder(Y.to(device), c.to(device))
    # predictions from the decoder
    Y_pred = decoder(mu_z, c.to(device))

    # output to CPU
    mu_z, sigma_z = mu_z.cpu(), sigma_z.cpu()
    Y_pred = Y_pred.cpu()

# + [markdown] id="vunBdUouYzb3" colab_type="text"
# ### Correlation between the ground truth $z$ and the inferred $z$ values

# + id="ZWGs64lIlKlr" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="20741db6-90f4-42d6-f22a-2fb42e165317"
plt.scatter(z, mu_z)

# + [markdown] id="IJP-xxIIY8gk" colab_type="text"
# ### Visualising mappings from z to feature space

# + id="hp9sFzohTXKN" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="1c2bc88d-f187-49af-8f4b-29572a5f5c8e"
plt.scatter(mu_z, Y_pred[:, 0], c=c.reshape(-1))

# + id="aszLDuDTTgf2" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="586401fd-8781-4995-ffd1-2eb9717d2e42"
plt.scatter(mu_z, Y_pred[:, 1], c=c.reshape(-1))

# + [markdown] id="Gp0hgR8zYvZi" colab_type="text"
# ### Inferred sparsity masks

# + id="_gwwYB8bGDRu" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="274eac82-08d0-4e56-86df-5201079c4cf9"
with torch.no_grad():
    sparsity = decoder.get_feature_level_sparsity_probs().cpu()
    
plt.imshow(sparsity)
plt.colorbar()

# + id="J3moKj3XqOzf" colab_type="code" colab={}

