
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


wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','NeuralDec')
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
latent_dim = 1 # init value: 1

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
    nn.Linear(hidden_dim, 2)
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
                                 logging_freq_int=100 ,
                                 n_iter=500, 
                                 augmented_lagrangian_lr=0.1)

#sys.exit(0)
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

# First let's see if the integrals have converged sufficiently close to zero
def plot_integrals(integrals):
    n_rep = integrals.shape[0]
    n_iter = integrals.shape[1]
    time = np.arange(n_iter).reshape(-1, 1)
    time_mat = np.tile(time, [1, n_rep])

    plt.plot(time_mat, integrals.T, c="black", alpha=0.25)
    plt.ylim(-0.5, 0.5)
#time_mat 
# Out[127]: 
# array([[  0,   0,   0, ...,   0,   0,   0],
#        [  1,   1,   1, ...,   1,   1,   1],
#        [  2,   2,   2, ...,   2,   2,   2],
#        ...,
#        [247, 247, 247, ..., 247, 247, 247],
#        [248, 248, 248, ..., 248, 248, 248],
#        [249, 249, 249, ..., 249, 249, 249]])
# 
#integrals.T
# Out[128]: 
# array([[ 4.10594940e-01, -8.01267177e-02, -8.36649314e-02, ...,
#         -1.82452202e-01, -1.87420383e-01, -3.84040438e-02],
#        [-7.32309837e-03, -6.12996006e-03,  2.94387341e-03, ...,
#         -5.89166760e-01, -2.54806668e-01, -5.69379091e-01],
#        [-1.80212427e-02, -4.26629111e-02, -2.04300489e-02, ...,
#         -3.67478818e-01, -3.24394822e-01, -3.63186479e-01],
#        ...,
#        [-8.22336692e-03, -2.36550183e-03, -6.33564778e-03, ...,
#         -6.70694326e-19, -9.33885861e-21, -2.49898213e-10],
#        [-1.51790376e-03, -4.67026228e-04, -7.50360498e-03, ...,
#         -4.89231652e-21, -2.47674718e-21, -7.25328478e-11],
# 
plot_integrals(integrals)

out_file_name = os.path.join(outfolder, 'plot_integrals' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

# forward = mapping + sparsity mask
# without sparsity mapping= forward
#         value = self.mapping_z(z) # mapping = a NN
#         if self.has_feature_level_sparsity:
#             w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_z)
#             return w * value



#     def forward_z(self, z):
#     def forward_c(self, c):
#     def forward_cz(self, z, c):
#     def forward(self, z, c):
#         return self.intercept + self.forward_z(z) + self.forward_c(c) + self.forward_cz(z, c)


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


# ### Visualising mappings from z to feature space

# Y_pred.shape
# 700 x 5
# mu_z shape 700 x 1

assert Y_pred.shape[0] == N
assert Y_pred.shape[1] == nfeatures

for i in range(nfeatures):
    plt.scatter(x=mu_z, y=Y[:, i], c=c.reshape(-1))
    plt.ylim([-2.5, 2.5])
    plt.ylabel("Observed values")
    plt.xlabel("z")
    plt.title("Feature " + str(i+1) + " - Observed data")
    out_file_name = os.path.join(outfolder, 'mapping_z_to_feature'+str(i+1)+'_obs' + outsuffix + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    plt.scatter(x=mu_z, y=Y_pred[:, i], c=c.reshape(-1))
    plt.ylim([-2.5, 2.5])
    plt.xlabel("z")
    plt.title("Feature " + str(i+1) + " - ND")
    out_file_name = os.path.join(outfolder, 'mapping_z_to_feature'+str(i+1)+'_ND_pred' + outsuffix + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    plt.scatter(x=mu_z, y=Y_pred_z[:, i], c=c.reshape(-1))
    plt.ylim([-2.5, 2.5])
    plt.xlabel("z")
    plt.title("Feature " + str(i+1) + " - f(z)")
    out_file_name = os.path.join(outfolder, 'mapping_z_to_feature'+str(i+1)+'_fz_predz' + outsuffix + '.png')         
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    plt.scatter(x=mu_z, y=Y_pred_c[:, i], c=c.reshape(-1))
    plt.ylim([-2.5, 2.5])
    plt.xlabel("z")
    plt.title("Feature " + str(i+1) + " - f(c)")
    out_file_name = os.path.join(outfolder, 'mapping_z_to_feature'+str(i+1)+'_fc_predc' + outsuffix + '.png')         
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    plt.scatter(x=mu_z, y=Y_pred_cz[:, i], c=c.reshape(-1))
    plt.ylim([-2.5, 2.5])
    plt.xlabel("z")
    plt.title("Feature " + str(i+1) + " - f(cz)")
    out_file_name = os.path.join(outfolder, 'mapping_z_to_feature'+str(i+1)+'_fcz_predcz' + outsuffix + '.png')         
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()


# ### Inferred sparsity masks

with torch.no_grad():
    # returns the w_z=torch.sigmoid(self.qlogits_z) (for z, c and zc)
    sparsity = decoder.get_feature_level_sparsity_probs().cpu()
    sparsity.shape
    #Out[7]: torch.Size([5, 3])
    # w_z, w_c, w_cz for the 5 features 

sparsity.shape 
# 4,3
# => how if multiple c ??? 

assert sparsity.shape[0] == nfeatures
assert sparsity.shape[1] == latent_dim + n_covariates + 1 # ???????

plt.imshow(sparsity)
plt.colorbar()

out_file_name = os.path.join(outfolder, 'sparsity_masks' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

### added MZ
# 
expvar_dt = pd.DataFrame(decoder.fraction_of_variance_explained(z=mu_z, c=c).numpy(),
                         columns=['z', 'c', 'cz'])
expvar_dt *= 100

for i in range(expvar_dt.shape[0]):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=pd.DataFrame(expvar_dt.iloc[i,:]).T)
    plt.title("Feature " + str(i+1) + " - Fraction of variance explained")
    plt.ylabel("% variance explained")
    out_file_name = os.path.join(outfolder, 'fract_explained_var_feature_'+str(i+1) + outsuffix + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()

print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))
sys.exit(0)




# for a given feature, plot value across LS value


#####################################################
 
 
#col         = 'consumption_energy'#
#conditions  = [ df2[col] >= 400, (df2[col] < 400) & (df2[col]> 200), df2[col] <= 200 ]
#choices     = [ "high", 'medium', 'low' ]
  
# df2["energy_class"] = np.select(conditions, choices, default=np.nan)