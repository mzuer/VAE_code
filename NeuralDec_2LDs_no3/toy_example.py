# python nd_toy_example.py

import sys,os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime



start_time = str(datetime.datetime.now().time())


wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','NeuralDec_2LDs')
os.chdir(wd)

outfolder = "ND_TOY_EXAMPLE"
os.makedirs(outfolder, exist_ok=True)


#module_path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code'
module_path = os.path.join(wd, 'ND')


if module_path not in sys.path:
    sys.path.append(module_path)


import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from ND.encoder import cEncoder
from ND.decoder_ld2 import Decoder
from ND.CVAE_ld2 import CVAE
from ND.helpers import expand_grid

from torch.utils.data import TensorDataset, DataLoader

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

# + [markdown] id="FrNDWlC7Z8TV" colab_type="text"
# Choose device (i.e. CPU or GPU)

# + id="RBWe-V94Zvwp" colab_type="code" colab={}
device = "cpu"

# + [markdown] id="7YcLw4FuZ-vF" colab_type="text"
# Generate a synthetic data set (700 data points, 5 features)

# + id="vxcIxlcPiuPW" colab_type="code" colab={}
N = 700

# generate ground truth latent variable z and covariate c
z1 = Uniform(-2.0, 2.0).sample((N, 1))
z2 = Uniform(-1.5, 1.5).sample((N, 1))
c = Uniform(-2.0, 2.0).sample((N, 1))
noise_sd = 0.05

# generate five features
y1 = torch.exp(-z1**2) - 0.2*c + torch.tanh(z2)
y2 = torch.sin(z1) + 0.2*c + 0.7*torch.sin(z1)*(z1 > 0).float()*c
y3 = torch.tanh(z1) + 0.2*c*torch.exp(-z2)
y4 = 0.2*z1 + torch.tanh(c)
y5 = 0.1*z1 * 0.2*z2

Y = torch.cat([y1, y2, y3, y4, y5], dim=1)
Y += noise_sd * torch.randn_like(Y)

Y = (Y - Y.mean(axis=0, keepdim=True)) / Y.std(axis=0, keepdim=True)

data_dim = Y.shape[1]
n_covariates = 1
hidden_dim = 32

n_lds = 2

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
    nn.Linear(hidden_dim, n_lds*2) ## is it n_lds *2 ???
)
# n_lds * 2 => each has one z_mu and one z_sigma

encoder = cEncoder(z_dim=n_lds, mapping=encoder_mapping)

# + id="FUesg8PHQWxn" colab_type="code" colab={}
### DECOMPOSABLE DECODER

# grid needed for quadrature

grid_z1 = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)
grid_z2 = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)  # [15,1]
grid_c = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)
grid_cz1 = torch.cat(expand_grid(grid_z1, grid_c), dim=1).to(device)
grid_cz2 = torch.cat(expand_grid(grid_z2, grid_c), dim=1).to(device)
grid_z1z2 = torch.cat(expand_grid(grid_z2, grid_z1), dim=1).to(device)
## ?????? but how to extend the grid in 3d ???

grid_cz1z2 = torch.cat(expand_grid(grid_z2, grid_z1, grid_c), dim=1).to(device) #### ???! not sure it will work with 3 terms

decoder_z1 = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)
decoder_z2 = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder_c = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder_z1z2 = nn.Sequential(
    nn.Linear(2, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)
decoder_cz1 = nn.Sequential(
    nn.Linear(2, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)
decoder_cz2 = nn.Sequential(
    nn.Linear(2, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder_cz1z2 = nn.Sequential(
    nn.Linear(3, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder = Decoder(data_dim, 
                 grid_z1=grid_z1, grid_z2=grid_z2, grid_c=grid_c, 
                 grid_cz1=grid_cz1, grid_cz2=grid_cz2, grid_z1z2=grid_z1z2,
                 grid_cz1z2=grid_cz1z2,
                 mapping_z1=decoder_z1,  mapping_z2=decoder_z2, mapping_c=decoder_c,
                 mapping_cz1=decoder_cz1, mapping_cz2=decoder_cz2, 
                 mapping_z1z2=decoder_z1z2, mapping_cz1z2=decoder_cz1z2,
                  has_feature_level_sparsity=True, 
                  p1=0.1, p2=0.1, p3=0.1, p4=0.1,
                  p5=0.1, p6=0.1, p7=0.1, 
                  lambda0=1e2, penalty_type="MDMM",
                  device=device)

# + [markdown] id="DB9uqHSVZgT_" colab_type="text"
# Combine the encoder + decoder and fit the decomposable CVAE

# + id="T-oOa_T2QYKd" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 907} outputId="02f18402-68d7-4685-e524-e2d2a8eeb23d"
model = CVAE(encoder, decoder, lr=5e-3, device=device)

loss, integrals = model.optimize(data_loader,
                                 n_iter=25000, 
                                 augmented_lagrangian_lr=0.1)
sys.exit(0)

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

