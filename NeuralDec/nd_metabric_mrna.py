
import sys,os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

inputfile= os.path.join('/home','marie','Documents','FREITAS_LAB',
                            'VAE_tutos','CancerAI-IntegrativeVAEs',
                            'data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv')

wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','NeuralDec')
os.chdir(wd)

#module_path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code'
module_path = os.path.join(wd, 'ND')


outfolder = "ND_METABRIC_MRNA"
os.makedirs(outfolder, exist_ok=True)


if module_path not in sys.path:
    sys.path.append(module_path)

from ND.encoder import cEncoder
from ND.decoder import Decoder
from ND.CVAE import CVAE
from ND.helpers import expand_grid

from torch.utils.data import TensorDataset, DataLoader

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

#####################
# load the data
###### load data

# training data
df=pd.read_csv(inputfile)

n_samp = df.shape[0]
n_genes = sum(['GE_' in x for x in df.columns])

mrna_data = df.iloc[:,34:1034].copy().values 
# the values after are CNA, the values before are clinical data
# copy() for deep copy
# values to convert to multidim array
mrna_data2= df.filter(regex='GE_').copy().values
assert mrna_data2.shape == mrna_data.shape

mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ \
(mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)

# will use the 'ER_Status' as condition

tokeep = np.where( (df['ER_Status'].values == "pos") | (df['ER_Status'].values == "neg"))
mrna_data_filt = mrna_data_scaled[list(tokeep[0]),:]

assert mrna_data_filt.shape[0] + df[df['ER_Status'] == "?"].shape[0] == mrna_data_scaled.shape[0]

input_data = mrna_data_filt
c_data = df['ER_Status'][list(tokeep[0])]
assert len(c_data) == mrna_data_filt.shape[0]
c_data_bin = np.vectorize(np.int)(c_data == "pos")
#c_data_bin = np.vectorize(np.float)(c_data == "pos")
assert np.all((c_data_bin == 1) | (c_data_bin==0))


# Choose device (i.e. CPU or GPU)
device = "cpu"


N = mrna_data_filt.shape[0]

####
#input_data=np.vectorize(float)(input_data)
c = torch.from_numpy(c_data_bin).reshape(-1,1).float()
Y = torch.from_numpy(input_data).float()
data_dim = Y.shape[1]
n_covariates = 1
hidden_dim = 32 # init value: 32
latent_dim = 1 # init value: 1

n_iter_integrals = 2000 # init value 25000 
logging_freq_integrals = 50 # init value 100
grid_nsteps = 15 # init value 15

#dataset = TensorDataset(Y.to(device), c.to(device))
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
encoder = cEncoder(z_dim=latent_dim, mapping=encoder_mapping)

### DECOMPOSABLE DECODER
grid_z = torch.linspace(-2.0, 2.0, steps=grid_nsteps).reshape(-1, 1).to(device)
grid_c = torch.linspace(-2.0, 2.0, steps=grid_nsteps).reshape(-1, 1).to(device)
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

plot_integrals(integrals)

out_file_name = os.path.join(outfolder, 'plot_integrals.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

# Now let's look at the inferred $z$ values, together with the mappings $z \mapsto \text{features}$
with torch.no_grad():
    # encoding of the entire observed data set
    mu_z, sigma_z = encoder(Y.to(device), c.to(device))
    # predictions from the decoder
    Y_pred = decoder(mu_z, c.to(device))

    # output to CPU
    mu_z, sigma_z = mu_z.cpu(), sigma_z.cpu()
    Y_pred = Y_pred.cpu()

# ### Visualising mappings from z to feature space
# Y_pred.shape
# 700 x 5
# mu_z shape 700 x 1

assert Y_pred.shape[0] == N
assert Y_pred.shape[1] == nfeatures

for i in range(5):
    
    plt.scatter(x=mu_z, y=Y_pred[:, i], c=c.reshape(-1))
    out_file_name = os.path.join(outfolder, 'mapping_z_to_feature'+str(i+1)+'.png')
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

out_file_name = os.path.join(outfolder, 'sparsity_masks.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

### added MZ
# from     def fraction_of_variance_explained(self, z, c, account_for_noise=False, divide_by_total_var=True):
# z and c are feed forward
expvar_dt = decoder.fraction_of_variance_explained(z=z, c=c)
expvar_dt.shape
# 4,3

# for a given feature, plot value across LS value


#####################################################
 
 
#col         = 'consumption_energy'#
#conditions  = [ df2[col] >= 400, (df2[col] < 400) & (df2[col]> 200), df2[col] <= 200 ]
#choices     = [ "high", 'medium', 'low' ]
  
# df2["energy_class"] = np.select(conditions, choices, default=np.nan)