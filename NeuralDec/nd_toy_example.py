
import sys,os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','NeuralDec')
os.chdir(wd)

#module_path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code'
module_path = os.path.join(wd, 'ND')


if module_path not in sys.path:
    sys.path.append(module_path)

from ND.encoder import cEncoder
from ND.decoder import Decoder
from ND.CVAE import CVAE
from ND.helpers import expand_grid

from torch.utils.data import TensorDataset, DataLoader

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

# Choose device (i.e. CPU or GPU)
device = "cpu"

# Generate a synthetic data set (700 data points, 5 features)
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

# torch.cat Concatenates the given sequence of seq tensors in the given dimension
Y = torch.cat([y1, y2, y3, y4, y5], dim=1)
# randn_like  Returns a tensor with the same size as input that is filled with random 
# numbers from a normal distribution with mean 0 and variance 1. 
Y += noise_sd * torch.randn_like(Y)

Y = (Y - Y.mean(axis=0, keepdim=True)) / Y.std(axis=0, keepdim=True)

data_dim = Y.shape[1]
n_covariates = 1
hidden_dim = 32

# torch to: Returns a Tensor with the specified device and (optional) dtype.
# For model fitting, we will need a `DataLoader` object
dataset = TensorDataset(Y.to(device), c.to(device))
data_loader = DataLoader(dataset, shuffle=True, batch_size=64)

# Setting up the CVAE encoder + decoder

### ENCODER

# define encoder which maps (data, covariate) -> (z_mu, z_sigma)
encoder_mapping = nn.Sequential(
    # torch.nn.Linear(size_in_features, size_out_features, ...)
    nn.Linear(data_dim + n_covariates, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 2)
)
#  Input: (N,∗,Hin) where ∗*∗ means any number of additional 
# dimensions and Hin=in_features
# Output: (N,∗,Hout) where all but the last dimension 
# are the same shape as the input and Hout=out_features

encoder = cEncoder(z_dim=1, mapping=encoder_mapping)

### DECOMPOSABLE DECODER

# grid needed for quadrature
# torch.linspace(start, end, steps, ...):
# Creates a one-dimensional tensor of size steps whose values are evenly
#  spaced from start to end, inclusive.
grid_z = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)
grid_c = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)
grid_cz = torch.cat(expand_grid(grid_z, grid_c), dim=1).to(device)

# grids are used in calculate_integrals

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
                  #mapping_z, mapping_c, mapping_cz
                  decoder_z, decoder_c, decoder_cz,
                  has_feature_level_sparsity=True, 
                  p1=0.1, p2=0.1, p3=0.1, # only needed if has_feature_level_sparsity
                  lambda0=1e2, penalty_type="MDMM",
                  device=device)

# Combine the encoder + decoder and fit the decomposable CVAE
model = CVAE(encoder, decoder, lr=5e-3, device=device)

loss, integrals = model.optimize(data_loader,
                                 n_iter=25000, 
                                 augmented_lagrangian_lr=0.1)


# ### Diagnostics and interpretation of the model fit

# First let's see if the integrals have converged sufficiently close to zero
def plot_integrals(integrals):
    n_rep = integrals.shape[0]
    n_iter = integrals.shape[1]
    time = np.arange(n_iter).reshape(-1, 1)
    time_mat = np.tile(time, [1, n_rep])

    plt.plot(time_mat, integrals.T, c="black", alpha=0.25)
    plt.ylim(-0.5, 0.5)

plot_integrals(integrals)

# Now let's look at the inferred $z$ values, together with the mappings $z \mapsto \text{features}$
with torch.no_grad():
    # encoding of the entire observed data set
    mu_z, sigma_z = encoder(Y.to(device), c.to(device))
    # predictions from the decoder
    Y_pred = decoder(mu_z, c.to(device))

    # output to CPU
    mu_z, sigma_z = mu_z.cpu(), sigma_z.cpu()
    Y_pred = Y_pred.cpu()

# ### Correlation between the ground truth $z$ and the inferred $z$ values
plt.scatter(z, mu_z)

# ### Visualising mappings from z to feature space

plt.scatter(mu_z, Y_pred[:, 0], c=c.reshape(-1))

plt.scatter(mu_z, Y_pred[:, 1], c=c.reshape(-1))

# ### Inferred sparsity masks

with torch.no_grad():
    # returns the w_z=torch.sigmoid(self.qlogits_z) (for z, c and zc)
    sparsity = decoder.get_feature_level_sparsity_probs().cpu()
    sparsity.shape
    #Out[7]: torch.Size([5, 3])
    # w_z, w_c, w_cz for the 5 features 
plt.imshow(sparsity)
plt.colorbar()

 ### added MZ
 decoder.fraction_of_variance_explained(z=z, c=c)
