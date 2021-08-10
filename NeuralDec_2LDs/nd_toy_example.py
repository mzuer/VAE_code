
###
# For this, you would first need to implement six "forward-mappings" and 
# then you could later calculate the variance explained for each of these mappings. 

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

from ND.encoder import cEncoder
from ND.decoder import Decoder
from ND.CVAE import CVAE
from ND.helpers import expand_grid

from torch.utils.data import TensorDataset, DataLoader

from torch.distributions.uniform import Uniform

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
#Y = torch.cat([y1, y2, y3, y4], dim=1)
# randn_like  Returns a tensor with the same size as input that is filled with random 
# numbers from a normal distribution with mean 0 and variance 1. 
Y += noise_sd * torch.randn_like(Y)

Y = (Y - Y.mean(axis=0, keepdim=True)) / Y.std(axis=0, keepdim=True)

data_dim = Y.shape[1]
n_covariates = 1
hidden_dim = 32 # init value: 32
latent_dim = 2 # init value: 1

n_iter_integrals = 200 # init value 25000 
logging_freq_integrals = 100 # init value 100
grid_nsteps = 15 # init value 15

outsuffix = "_hd" + str(hidden_dim) + "_nLD" + str(latent_dim) +\
   "_" + str(n_iter_integrals) + "_" + str(logging_freq_integrals) +"_"+str(grid_nsteps) 


# with 5 features, integral shape: 160,25  
# with 4 features, integral shape: 128,25
# nfeatures x 32; but changing hidden dim size does not affect
# 32 = 2*grid_nsteps+2
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
    nn.Linear(hidden_dim, latent_dim*2)  ### !! change here, for each LD return z_mu and z_sigma (init value:2)
)
#  Input: (N,∗,Hin) where ∗*∗ means any number of additional 
# dimensions and Hin=in_features
# Output: (N,∗,Hout) where all but the last dimension 
# are the same shape as the input and Hout=out_features

encoder = cEncoder(z_dim=latent_dim, mapping=encoder_mapping)

### DECOMPOSABLE DECODER

# grid needed for quadrature
# torch.linspace(start, end, steps, ...):
# Creates a one-dimensional tensor of size steps whose values are evenly
#  spaced from start to end, inclusive.
grid_z = torch.linspace(-2.0, 2.0, steps=grid_nsteps).reshape(-1, 1).to(device)
grid_c = torch.linspace(-2.0, 2.0, steps=grid_nsteps).reshape(-1, 1).to(device)
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

## integrals come from:
### collect all integral values into one array
###integrals = np.hstack([int_z_values, int_c_values, int_cz_values]).reshape(n_iter // logging_freq_int, -1).T
## logging_freq_int default = 100

#n_iter/logging_freq_int => gives the # of columns
loss, integrals = model.optimize(data_loader,
                                 logging_freq_int=logging_freq_integrals ,
                                 n_iter=n_iter_integrals, 
                                 augmented_lagrangian_lr=0.1)

sys.exit(0)








loss.shape
# 1250,
integrals.shape
# 160,250
# => the columns of integrals = number of iterations divided by the frequency of logging
# => the rows: at each logging step, nbr features x nbr hidden dim

nfeatures = Y.shape[1]
 
assert integrals.shape[0] == nfeatures * (grid_nsteps * 2 + 2)
assert integrals.shape[1] == (n_iter_integrals//logging_freq_integrals)

# ### Diagnostics and interpretation of the model fit
sys.exit(0)

# First let's see if the integrals have converged sufficiently close to zero
def plot_integrals(integrals):
    n_rep = integrals.shape[0]
    n_iter = integrals.shape[1]
    time = np.arange(n_iter).reshape(-1, 1)
    time_mat = np.tile(time, [1, n_rep])

    plt.plot(time_mat, integrals.T, c="black", alpha=0.25)
    plt.ylim(-0.5, 0.5)
#time_mat 
# 
plot_integrals(integrals)

out_file_name = os.path.join(outfolder, 'plot_integrals' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()




# Now let's look at the inferred $z$ values, together with the mappings $z \mapsto \text{features}$
with torch.no_grad():
    # encoding of the entire observed data set
    mu_z, sigma_z = encoder(Y.to(device), c.to(device))
    # predictions from the decoder
    Y_pred = decoder(mu_z, c.to(device))
    Y_pred_c = decoder.forward_c(c.to(device))
    Y_pred_cz = decoder.forward_cz(mu_z, c.to(device))
    Y_pred_z = decoder.forward_z(mu_z)

    # output to CPU
    mu_z, sigma_z = mu_z.cpu(), sigma_z.cpu()
    Y_pred = Y_pred.cpu()
    Y_pred_c = Y_pred_c.cpu()
    Y_pred_cz = Y_pred_cz.cpu()
    Y_pred_z = Y_pred_z.cpu()


# because this is synthetic data -> we have the true generative factor
# ### Correlation between the ground truth $z$ and the inferred $z$ values
plt.scatter(z, mu_z)
out_file_name = os.path.join(outfolder, 'correlation_true_inferred' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

file   = open(os.path.join("ND_TOY_EXAMPLE/data_subset_debug_optimize.sav"), 'rb')
data_subset = pickle.load(file)

file   = open(os.path.join("ND_TOY_EXAMPLE/out_debug_encoder.sav"), 'rb')
out = pickle.load(file)


############### trial increasing # latent dim
# latent_dim = 2
    z = mu_z + sigma_z * eps

RuntimeError: The size of tensor a (0) must match the size of tensor b (2) at non-singleton dimension 1
### come from 
                loss, int_z, int_c, int_cz_dc, int_cz_dz = self.forward(data_subset, beta=1.0, device=self.device)
model.forward(data_subset, beta=1.0, device="cpu")

# data_subset -> split data for batch on epoch (row wise)
# 1st element is Y 64x5 [# of subsmaples x # of features]
# 2nd is c, [# of subsamp, and # of features]

Y, c = data_subset
Y, c = Y.to(device), c.to(device)
mu_z, sigma_z = encoder(Y, c)
# mu_z 64,2
# sigma_z 64,0 ??


#                loss, int_z, int_c, int_cz_dc, int_cz_dz = self.forward(data_subset, beta=1.0, device=self.device)
# ligne dans CVAE qui appelle le forward de CVAE
 
# encode
mu_z, sigma_z = model.encoder(Y, c)
eps = torch.randn_like(mu_z)
z = mu_z + sigma_z * # eps  <<<< erreur de cette ligne là
## => il y a porlbème dans le sigma -> aller voir dans l'encodeur

# mu_z 64,2 
# sigma_z 64,0
assert list(mu_z.shape) == list(sigma_z.shape) # wrong -> so there is an issue in the encoder
# ISSUE here -> mu_z and sigma_z must be of same shape

eps = torch.randn_like(mu_z)
z = mu_z + sigma_z * eps



    def forward(self, data_subset, beta=1.0, device="cpu"):
        # we assume data_subset containts two elements
        Y, c = data_subset
        Y, c = Y.to(device), c.to(device)

        # encode
        mu_z, sigma_z = self.encoder(Y, c)
        eps = torch.randn_like(mu_z)
        z = mu_z + sigma_z * eps

        # decode
        y_pred = self.decoder.forward(z, c)
        decoder_loss, penalty, int1, int2, int3, int4 = self.decoder.loss(y_pred, Y)

        # loss function
        VAE_KL_loss = KL_standard_normal(mu_z, sigma_z)

        # Note that when this loss (neg ELBO) is calculated on a subset (minibatch),
        # we should scale it by data_size/minibatch_size, but it would apply to all terms
        total_loss = decoder_loss + beta * VAE_KL_loss

        return total_loss, int1, int2, int3, int4






def init_forward(self, data_subset, beta=1.0, device="cpu"):
    # we assume data_subset containts two elements
    Y, c = data_subset
    Y, c = Y.to(device), c.to(device)

    # encode
    mu_z, sigma_z = self.encoder(Y, c)
    eps = torch.randn_like(mu_z)
    z = mu_z + sigma_z * eps

    # decode
    y_pred = self.decoder.forward(z, c)
    decoder_loss, penalty, int1, int2, int3, int4 = self.decoder.loss(y_pred, Y)

    # loss function
    VAE_KL_loss = KL_standard_normal(mu_z, sigma_z)

    # Note that when this loss (neg ELBO) is calculated on a subset (minibatch),
    # we should scale it by data_size/minibatch_size, but it would apply to all terms
    total_loss = decoder_loss + beta * VAE_KL_loss

    return total_loss, int1, int2, int3, int4



model.init_forward(data_subset)



z_dim=2
mapping=encoder_mapping


class cEncoder(nn.Module):
     """
    Encoder module for CVAE,
    i.e. it maps (Y, c) to the approximate posterior q(z)=N(mu_z, sigma_z)
    """

    def __init__(self, z_dim, mapping):
        super().__init__()

        self.z_dim = z_dim

        # NN mapping from (Y, x) to z
        self.mapping = mapping

    def forward(self, Y, c):

        out = self.mapping(torch.cat([Y, c], dim=1)) # give Y and C to the NN

        mu = out[:, 0:self.z_dim]
        # SoftPlus is a smooth approximation to the ReLU function and can 
        # be used to constrain the output of a machine to always be positive.
        sigma = 1e-6 + softplus(out[:, self.z_dim:(2 * self.z_dim)])

        return mu, sigma


