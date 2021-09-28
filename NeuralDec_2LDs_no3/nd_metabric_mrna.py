
# python nd_metabric_mrna.py

import sys,os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import random
import datetime
import seaborn as sns
from scipy.stats import spearmanr
from pingouin import mwu

start_time = str(datetime.datetime.now().time())

random.seed(123)


import matplotlib.pyplot as plt

inputfile= os.path.join('/home','marie','Documents','FREITAS_LAB',
                            'VAE_tutos','CancerAI-IntegrativeVAEs',
                            'data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv')

wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','NeuralDec_2LDs_no3')
os.chdir(wd)

#module_path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code'
module_path = os.path.join(wd, 'ND')


outfolder = "ND_METABRIC_MRNA"
os.makedirs(outfolder, exist_ok=True)


if module_path not in sys.path:
    sys.path.append(module_path)

from ND.encoder import cEncoder
from ND.decoder_ld2 import Decoder
from ND.CVAE_ld2 import CVAE
from ND.helpers import expand_grid

from torch.utils.data import TensorDataset, DataLoader

#from torch.distributions.uniform import Uniform
#from torch.distributions.normal import Normal


#runModel = True

#####################
# load the data
###### load data

covarLab = "ER_Status"

# training data
df=pd.read_csv(inputfile)

n_samp = df.shape[0]
n_genes = sum(['GE_' in x for x in df.columns])

mrna_data = df.iloc[:,34:1034].copy().values 
gene_names = [re.sub('GE_', '', x) for x in df.columns[34:1034]]
# the values after are CNA, the values before are clinical data
# copy() for deep copy
# values to convert to multidim array
mrna_data2= df.filter(regex='GE_').copy().values
assert mrna_data2.shape == mrna_data.shape

mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ \
(mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)

# will use the 'ER_Status' as condition
tokeep = np.where( (df[covarLab].values == "pos") | (df[covarLab].values == "neg"))
mrna_data_filt = mrna_data_scaled[list(tokeep[0]),:]

sample_labels = df[covarLab].values[tokeep]

assert mrna_data_filt.shape[0] + df[df[covarLab] == "?"].shape[0] == mrna_data_scaled.shape[0]

input_data = mrna_data_filt
c_data = df[covarLab][list(tokeep[0])]
assert len(c_data) == mrna_data_filt.shape[0]
c_data_bin = np.vectorize(np.int)(c_data == "pos")
#c_data_bin = np.vectorize(np.float)(c_data == "pos")
assert np.all((c_data_bin == 1) | (c_data_bin==0))


# Choose device (i.e. CPU or GPU)
device = "cpu"

N = input_data.shape[0]
nsamp = N
assert len(sample_labels) == nsamp
####

# casting needed:
# type(input_data[0,0])
# Out[161]: numpy.float64
# 
# type(Y.numpy()[0,0])
# Out[162]: numpy.float32
# 

#input_data=np.vectorize(float)(input_data)
c = torch.from_numpy(c_data_bin).reshape(-1,1).float() ### ?? requires float input
Y = torch.from_numpy(input_data).float() ### ?? requires float input
data_dim = Y.shape[1]
n_covariates = 1
hidden_dim = 32 # init value: 32
latent_dim = 2 # init value: 1

n_iter_integrals = 100 # init value 25000 
logging_freq_integrals = 50 # init value 100
grid_nsteps = 15 # init value 15

bs = 64 # init value 64 (batch size)

min_grid_range = -2 # init value -2
max_grid_range = 2 # init value -2

outsuffix = "_hd" + str(hidden_dim) + "_nLD" + str(latent_dim) +\
 "_c" + str(covarLab) +  "_" + str(n_iter_integrals) + "_" + str(logging_freq_integrals) +"_"+str(grid_nsteps) 

#dataset = TensorDataset(Y.to(device), c.to(device))
dataset = TensorDataset(Y.to(device), c.to(device))
data_loader = DataLoader(dataset, shuffle=True, batch_size=bs)

# Setting up the CVAE encoder + decoder

### ENCODER

# define encoder which maps (data, covariate) -> (z_mu, z_sigma)
encoder_mapping = nn.Sequential(
    # torch.nn.Linear(size_in_features, size_out_features, ...)
    nn.Linear(data_dim + n_covariates, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, latent_dim*2)
)
encoder = cEncoder(z_dim=latent_dim, mapping=encoder_mapping)

### DECOMPOSABLE DECODER

grid_z1 = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)
grid_z2 = torch.linspace(-2.0, 2.0, steps=15).reshape(-1, 1).to(device)  # [15,1]
grid_c = torch.linspace(0, 1, steps=15).reshape(-1, 1).to(device)  # =>> if binary, 0-1
grid_cz1 = torch.cat(expand_grid(grid_z1, grid_c), dim=1).to(device) # :0 is range of z; :1 is range of c
grid_cz2 = torch.cat(expand_grid(grid_z2, grid_c), dim=1).to(device)
grid_z1z2 = torch.cat(expand_grid(grid_z2, grid_z1), dim=1).to(device)
## ?????? but how to extend the grid in 3d ???


decoder_z1 = nn.Sequential(   #### z1 should be feed only with z1 ??!
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


decoder = Decoder(data_dim, 
                 grid_z1=grid_z1, grid_z2=grid_z2, grid_c=grid_c, 
                 grid_cz1=grid_cz1, grid_cz2=grid_cz2, grid_z1z2=grid_z1z2,
                 mapping_z1=decoder_z1,  mapping_z2=decoder_z2, mapping_c=decoder_c,
                 mapping_cz1=decoder_cz1, mapping_cz2=decoder_cz2, 
                 mapping_z1z2=decoder_z1z2,
                  has_feature_level_sparsity=True, 
                  p1=0.1, p2=0.1, p3=0.1, p4=0.1,
                  p5=0.1, p6=0.1, p7=0.1, 
                  lambda0=1e2, penalty_type="MDMM",
                  device=device)

# Combine the encoder + decoder and fit the decomposable CVAE
#model = CVAE(encoder, decoder, lr=5e-3, device=device)
model = CVAE(encoder, decoder, lr=1e-3, device=device)

## integrals come from:
### collect all integral values into one array
###integrals = np.hstack([int_z_values, int_c_values, int_cz_values]).reshape(n_iter // logging_freq_int, -1).T
## logging_freq_int default = 100

#n_iter/logging_freq_int => gives the # of columns
loss, integrals = model.optimize(data_loader,
                                 logging_freq_int=logging_freq_integrals ,
                                 n_iter=n_iter_integrals, 
                                 augmented_lagrangian_lr=0.1)


outfile = os.path.join(outfolder, "vae_model" + outsuffix + "_dict.pt")
torch.save(model.state_dict(), outfile)
print("... written: " + outfile)
outfile = os.path.join(outfolder, "vae_model" + outsuffix + "_full.pt")
torch.save(model, outfile)
print("... written: " + outfile)

loss.shape
# 1250,
integrals.shape
# 160,250
# => the columns of integrals = number of iterations divided by the frequency of logging
# => the rows: at each logging step, nbr features x nbr hidden dim

nfeatures = Y.shape[1]
 
assert integrals.shape[0] == nfeatures * (grid_nsteps * 3 * 2 + 3*1) # was *2 + 2 ###  

# à chaque step les intégrales ont la forme suivante:
# pour les grid simples (e.g. c, z1, z2) -> je passe les 15 points dans le decoder [15,1]
# ca me donne un output [1000] reshape en 1 x [1,1000]
# pour les grids doubles (e.g. cz1, cz2, z1z2) -> c est en 2d on a fait un quadrillage
# donc la grille est de [15x15=225,2] -> ca donne un output [15x15=225,1000]
# je reshape en [15,15,1000] (pour chaque feature, j'ai un quadrillage)
# aggregation en dim 0 et dim 1 (e.g. une fois pour dc, une fois pour dz1) => ca donne 
# 2x [15,1000] (NB: une aggreg en dim3 donnerait un result de 15x15)
# donc à chaque step 
# nbr_decoders_1var * 1 + 15*2*nbr_decoders_2var
# comme stacké -> * nfeatures
 
#  2* # 2d grid + # 1d grid  cz1,cz2,z1z2=>2d => 2*3 + c, z1, z2 => 1*3               * # networks * 2 + # networks  # or because 2d grid 
assert integrals.shape[1] == (n_iter_integrals//logging_freq_integrals)


#        integrals = np.hstack([int_z1_values, int_z2_values, int_c_values,\
#                               int_cz1_values, int_cz2_values,int_z1z2_values ]).\
#                                    reshape(n_iter // logging_freq_int, -1).T

#sys.exit(0)


# ### Diagnostics and interpretation of the model fit
out_file_name = os.path.join(outfolder, 'model_loss' + outsuffix + '.png')
sns.lineplot(y=loss,x=range(len(loss)), linestyle='-', marker='o')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

# First let's see if the integrals have converged sufficiently close to zero
def plot_integrals(integrals):
    n_rep = integrals.shape[0]
    n_iter = integrals.shape[1]
    time = np.arange(n_iter).reshape(-1, 1)
    time_mat = np.tile(time, [1, n_rep])

    plt.plot(time_mat, integrals.T, c="black", alpha=0.25)
    plt.ylim(-0.5, 0.5)

plot_integrals(integrals)
out_file_name = os.path.join(outfolder, 'plot_integrals' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()






# + id="UEVeue2dzJfZ" colab_type="code" colab={}
with torch.no_grad():
    # encoding of the entire observed data set
    mu_z, sigma_z = encoder(Y.to(device), c.to(device))
    mu_z1 = mu_z[:,0].reshape(-1,1)
    mu_z2 = mu_z[:,1].reshape(-1,1)
    sigma_z1 = sigma_z[:,0].reshape(-1,1)
    sigma_z2 = sigma_z[:,1].reshape(-1,1)
    # predictions from the decoder
    Y_pred = decoder(mu_z1.to(device), mu_z2.to(device), c.to(device))

    # output to CPU
    mu_z, sigma_z = mu_z.cpu(), sigma_z.cpu()
    mu_z1, sigma_z1 = mu_z1.cpu(), sigma_z1.cpu()
    mu_z2, sigma_z2 = mu_z2.cpu(), sigma_z2.cpu()
    Y_pred = Y_pred.cpu()

# + [markdown] id="vunBdUouYzb3" colab_type="text"
# ### Correlation between the ground truth $z$ and the inferred $z$ values

# + id="ZWGs64lIlKlr" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="20741db6-90f4-42d6-f22a-2fb42e165317"
#¨plt.scatter(z, mu_z)

# + [markdown] id="IJP-xxIIY8gk" colab_type="text"
# ### Visualising mappings from z to feature space

# + id="hp9sFzohTXKN" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="1c2bc88d-f187-49af-8f4b-29572a5f5c8e"
plt.scatter(mu_z2, Y_pred[:, 0], c=c.reshape(-1))
plt.scatter(mu_z1, Y_pred[:, 0], c=c.reshape(-1))





























sys.exit(0)


# Now let's look at the inferred $z$ values, together with the mappings $z \mapsto \text{features}$
with torch.no_grad():
    # encoding of the entire observed data set
    mu_z, sigma_z = encoder(Y.to(device), c.to(device))
    # predictions from the decoder
    Y_pred = decoder(mu_z, c.to(device))
    
    # get also decomposition
    Y_pred_c = decoder.forward_c(c.to(device))
    Y_pred_cz = decoder.forward_cz(mu_z, c.to(device))
    Y_pred_z = decoder.forward_z(mu_z)

    # output to CPU
    mu_z, sigma_z = mu_z.cpu(), sigma_z.cpu()
    Y_pred = Y_pred.cpu()
    Y_pred_c = Y_pred_c.cpu()
    Y_pred_cz = Y_pred_cz.cpu()
    Y_pred_z = Y_pred_z.cpu()

    
assert Y_pred.shape[0] == Y.shape[0]
assert Y_pred.shape[1] == Y.shape[1]
assert Y_pred.shape[0] == nsamp
assert Y_pred.shape[1] == nfeatures
assert mu_z.shape[1] == latent_dim
assert mu_z.shape[0] == nsamp
assert Y_pred_c.shape[0] == nsamp
assert Y_pred_c.shape[1] == nfeatures
assert Y_pred_cz.shape[0] == nsamp
assert Y_pred_cz.shape[1] == nfeatures
assert Y_pred_z.shape[0] == nsamp
assert Y_pred_z.shape[1] == nfeatures



# ### Inferred sparsity masks
with torch.no_grad():
    # returns the w_z=torch.sigmoid(self.qlogits_z) (for z, c and zc)
    sparsity = decoder.get_feature_level_sparsity_probs().cpu()
    sparsity.shape
    #Out[7]: torch.Size([5, 3])
    # w_z, w_c, w_cz for the 5 features 
sparsity.shape 
# 1000,3
# => how if multiple c ??? 
assert sparsity.shape[0] == nfeatures
assert sparsity.shape[1] == latent_dim + n_covariates + 1 # ???????

### correlation with LD for each feature
# there is only one for the moment ?
# for each gene I have expression for all samples
# that I can correlate with the LD values (1 per sample)

latent_repr = np.array(mu_z)

correlations_all=[]
p_values_all=[]
for gene_i in range(nfeatures):
    correlations=[]
    p_values=[]
    for latent_dim_i in range(latent_dim):
        corr_, p_value = spearmanr(input_data[:,gene_i], latent_repr[:,latent_dim_i])
        correlations.append(corr_)
        p_values.append(p_value)
    correlations_all.append(correlations)
    p_values_all.append(p_values)

correlations_all = np.array(correlations_all)
correlations_all_df = pd.DataFrame(correlations_all.T, columns = gene_names)
p_values_all = np.array(p_values_all)
p_values_all_df  = pd.DataFrame(p_values_all.T, columns = gene_names)


####****************************** VISUALIZATION

# ### Visualising mappings from z to feature space
# Y_pred.shape
# nsamp x nfeatures
# mu_z shape nsamp x n_latent_dim]

#assert np.all(np.equal(Y.numpy(), input_data)) ### False ?????????????????????
#np.float(Y.numpy()[0,0]) == np.float(input_data[0,0]) ### False ????????
# np.float(Y.numpy()[0,0]) ????????????
# Out[148]: 0.17122244834899902
# np.float(input_data[0,0]) ???????????????????????????????????
# Out[149]: 0.17122244293147418
### because casting to float ???

# np.round(input_data[0,0], 4)
# Out[153]: 0.1712
# 
# np.round(Y.numpy()[0,0], 4)
# Out[154]: 0.1712
# 
# np.round(input_data[0,0], 4)==np.round(Y.numpy()[0,0], 4)
# Out[155]: False
# 
# np.float(np.round(input_data[0,0], 4))==np.float(np.round(Y.numpy()[0,0], 4))
# Out[156]: False
# 
# np.double(np.round(input_data[0,0], 4))==np.double(np.round(Y.numpy()[0,0], 4))
# Out[157]: False

assert np.all(np.equal(Y.numpy().astype(np.float32), input_data.astype(np.float32)))

assert Y_pred.shape[1] == len(gene_names)

ntopvar_genes_toplot = 3

ntop_genes_toplot = 5

plt_hue = sample_labels

assert np.all(np.equal(latent_repr, np.array(mu_z)))


# for the moment, only 1 LD
for i_ld in range(latent_dim):
    i_latent_repr = latent_repr[:,i_ld]
    assert i_latent_repr.size == nsamp

    for i_g in range(ntop_genes_toplot):
    
        i_ypred = np.array(Y_pred[:,i_g])
        i_y = np.array(Y[:,i_g])
        
        xlab = "LD "+str(i_ld+1)

        ### Y vs mu_z
        corr_, p_value = spearmanr(i_latent_repr, i_y)
        fig, ax = plt.subplots(figsize=(6,6))
        sns.scatterplot(x=i_latent_repr, y=i_y, hue=plt_hue)
        plt.title(gene_names[i_g] + " (SCC={:.4f} - pval={:.2e})".format(corr_, p_value))
        plt.xlabel(xlab)
        plt.ylabel("Gene expr (gene " + str(i_g+1) + "=" +gene_names[i_g] +")")
        out_file_name = os.path.join(outfolder, 'Y_vs_mu_z_LD' + str(i_ld+1)+'_feature'+str(i_g+1) + outsuffix +'.png')
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()
    
        ### Y_pred vs mu_z
        corr_, p_value = spearmanr(i_latent_repr, i_ypred)
        fig, ax = plt.subplots(figsize=(6,6))
        sns.scatterplot(x=i_latent_repr, y=i_ypred, hue=plt_hue)
        plt.title(gene_names[i_g] + " (SCC={:.4f} - pval={:.2e})".format(corr_, p_value))
        plt.xlabel(xlab)
        plt.ylabel("Pred. gene expr (gene " + str(i_g+1) + "=" +gene_names[i_g] +")")
        out_file_name = os.path.join(outfolder, 'Ypred_vs_mu_z_LD' + str(i_ld+1)+'_feature'+str(i_g+1) + outsuffix +'.png')
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()
        
        ### Y_pred vs Y
        corr_, p_value = spearmanr(i_y, i_ypred)
        fig, ax = plt.subplots(figsize=(6,6))
        sns.scatterplot(x=i_y, y=i_ypred, hue=plt_hue)
        plt.title(gene_names[i_g] + " (SCC={:.4f} - pval={:.2e})".format(corr_, p_value))
        plt.scatter(x=Y[:,i_g], y=Y_pred[:, i_g], c=c.reshape(-1))
        plt.title(gene_names[i_g])
        plt.xlabel("Gene expr (gene " + str(i_g+1) + "=" +gene_names[i_g] +")")
        plt.ylabel("Pred. gene expr (gene " + str(i_g+1) + "=" +gene_names[i_g] +")")
        out_file_name = os.path.join(outfolder, 'Ypred_vs_Y_LD' + str(i_ld+1)+'_feature'+str(i_g+1) + outsuffix +'.png')
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()


# sparsity returns w_z, w_c, w_cz
#return torch.cat([w_z, w_c, w_cz], dim=0).t()
sparsity_df = pd.DataFrame(sparsity.numpy())
sparsity_df_sorted = sparsity_df.sort_values(by=0, axis=0, ascending=False)
fig, ax = plt.subplots(figsize=(6,6))
p=ax.imshow(sparsity_df_sorted)
ax.set_aspect('auto')
plt.title("sparsity mask [sorted by z col.]")
plt.xlabel("sparsity masks: z - c - cz")
plt.ylabel("Features (genes)")
plt.axvline(x=0.5, color="black")
plt.axvline(x=1.5, color="black")
fig.colorbar(p)

out_file_name = os.path.join(outfolder, 'sparsity_masks' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

# plot correlation vs. mask value
# # those that have high value in sparsity mask for c 
# should have low value of correlation

# correlation LD vs sparsity mask
# for the moment, only 1 LD

i_ld = 0
corr_with_LD = np.array(correlations_all_df)[i_ld,:]

i_SM = 1
sm_col = "c"
SM_c = np.array(sparsity_df)[:,i_SM]
fig, ax = plt.subplots(figsize=(6,6))
sns.scatterplot(x=SM_c , y=corr_with_LD)
plt.title("Correlation vs. sparsity mask (feature-level [gene-level])")
plt.xlabel("Sparsity mask for "+sm_col)
plt.ylabel("Correlation gene expr. with LD"+str(i_ld+1))
out_file_name = os.path.join(outfolder, 'sparsity_mask_for_'+sm_col+'_LD'+str(i_ld+1) + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()


i_SM = 0
sm_col = "z"
SM_c = np.array(sparsity_df)[:,i_SM]
fig, ax = plt.subplots(figsize=(6,6))
sns.scatterplot(x=SM_c , y=corr_with_LD)
plt.title("Correlation vs. sparsity mask (feature-level [gene-level])")
plt.xlabel("Sparsity mask for "+sm_col)
plt.ylabel("Correlation gene expr. with LD"+str(i_ld+1))
out_file_name = os.path.join(outfolder, 'sparsity_mask_for_'+sm_col+'_LD'+str(i_ld+1) + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

i_SM = 2
sm_col = "zc"
SM_c = np.array(sparsity_df)[:,i_SM]
fig, ax = plt.subplots(figsize=(6,6))
sns.scatterplot(x=SM_c , y=corr_with_LD)
plt.title("Correlation vs. sparsity mask (feature-level [gene-level])")
plt.xlabel("Sparsity mask for "+sm_col)
plt.ylabel("Correlation gene expr. with LD"+str(i_ld+1))
out_file_name = os.path.join(outfolder, 'sparsity_mask_for_'+sm_col+'_LD'+str(i_ld+1) + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()



# boxplot variance explained
#f_all_var = torch.cat([f_z_var, f_c_var, f_int_var], dim=0) return f_all_var.t()
expvar_dt = pd.DataFrame(decoder.fraction_of_variance_explained(z=mu_z, c=c).numpy(),
                         columns=['z', 'c', 'cz'])
expvar_dt *= 100

fig, ax = plt.subplots(figsize=(10,6))
sns.violinplot(data=expvar_dt, color="0.8")
sns.stripplot(data=expvar_dt, jitter=True, zorder=1)
plt.title("Fraction of variance explained")
plt.ylabel("% variance explained")
out_file_name = os.path.join(outfolder, 'fraction_of_variation_explained_dist' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

#sys.exit(0)


# for the top-ranking of variance explained, decompose the curve
tmp_dt = expvar_dt.copy()
ntopvar_genes_toplot = 3

# I need to retrieve the top genes for each of the variance
# for each variance (c, z, or cz) -> sort the table
# and retrieve the ntopvar_genes_toplot
# for each of this gene
# 1) do the barplot
# 2) plot the mapping ND, fz, fc, fcz
var_type = tmp_dt.columns[0] # plot the
i_top = 0

for var_type in tmp_dt.columns:
    tmp_dt['gene'] = gene_names
    tmp_dt['i_gene'] = range(len(gene_names))
    tmp2_dt = tmp_dt.sort_values(by=var_type, axis=0, ascending=False)
        
    for i_top in range(ntopvar_genes_toplot):
         
        i_gene_name = tmp2_dt.iloc[i_top,:]['gene']
        i_gene_idx = tmp2_dt.iloc[i_top,:]['i_gene']
        
        outsuffix_2 = "_" + i_gene_name + "_var_" + var_type + "_top" + str(i_top+1) + outsuffix

        # first the barplot

        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=pd.DataFrame(tmp2_dt.iloc[i_top,:][['z', 'c', 'cz']]).T)
        plt.title("Fraction of variance explained - " + i_gene_name)
        plt.ylabel("% variance explained")
        plt.xlabel("Top " + str(i_top+1) + " expl. var. " + var_type  + " - " + i_gene_name)
        out_file_name = os.path.join(outfolder, 'fract_explained_var'  + outsuffix_2 + '.png')
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()
        
        
        yaxmin= min(list(Y[:, i_gene_idx].numpy()) + list(Y_pred[:, i_gene_idx].numpy()) +
                    list(Y_pred_c[:, i_gene_idx].numpy()) +
                    list(Y_pred_z[:, i_gene_idx].numpy()) +
                    list(Y_pred_cz[:, i_gene_idx].numpy()) 
                    )
        yaxmax= max(list(Y[:, i_gene_idx].numpy()) + list(Y_pred[:, i_gene_idx].numpy()) +
                    list(Y_pred_c[:, i_gene_idx].numpy()) +
                    list(Y_pred_z[:, i_gene_idx].numpy()) +
                    list(Y_pred_cz[:, i_gene_idx].numpy()) 
                    )            

        
                # observed
        plt.scatter(x=mu_z, y=Y[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
        plt.ylim([yaxmin, yaxmax])
        plt.ylabel("Observed values")
        plt.xlabel("z")
        plt.title("Observed data " + i_gene_name + " (var "+var_type+" top" + str(i_top+1)+')')
        out_file_name = os.path.join(outfolder, 'mapping_z_to_obs' + outsuffix_2 + '.png')
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()
        # mapping ND
        plt.scatter(x=mu_z, y=Y_pred[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
        plt.ylim([yaxmin, yaxmax])
        plt.xlabel("z")
        plt.title("ND " + i_gene_name + " (var "+var_type+" top" + str(i_top+1)+')')
        out_file_name = os.path.join(outfolder, 'mapping_z_to_ND_pred' + outsuffix_2 + '.png')
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()
        # mapping fz
        plt.scatter(x=mu_z, y=Y_pred_z[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
        plt.ylim([yaxmin, yaxmax])
        plt.xlabel("z")
        plt.title("f(z) " + i_gene_name + " (var "+var_type+" top" + str(i_top+1)+')')
        out_file_name = os.path.join(outfolder, 'mapping_z_to_fz_predz' + outsuffix_2 + '.png')         
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()
        # mapping fc
        plt.scatter(x=mu_z, y=Y_pred_c[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
        plt.ylim([yaxmin, yaxmax])
        plt.xlabel("z")
        plt.title("f(c) " + i_gene_name + " (var "+var_type+" top" + str(i_top+1)+')')
        out_file_name = os.path.join(outfolder, 'mapping_z_to_fc_predc' + outsuffix_2 + '.png')         
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()
        # mapping fcz
        plt.scatter(x=mu_z, y=Y_pred_cz[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
        plt.ylim([yaxmin, yaxmax])
        plt.xlabel("z")
        plt.title("f(cz) " + i_gene_name + " (var "+var_type+" top" + str(i_top+1)+')')
        out_file_name = os.path.join(outfolder, 'mapping_z_to_fcz_predcz' + outsuffix_2 + '.png')         
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()
                
    
rd_idxs = random.sample(range(0, nfeatures), ntopvar_genes_toplot)
for rd_i, i_gene_idx in enumerate(rd_idxs):
    
    i_gene_name = gene_names[i_gene_idx]
    i_top = rd_i
    outsuffix_2 = "_" + i_gene_name + "_random_" + str(i_top+1) + outsuffix

    # first the barplot
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=pd.DataFrame(tmp_dt.iloc[i_top,:][['z', 'c', 'cz']]).T)
    plt.title("Fraction of variance explained - " + i_gene_name)
    plt.ylabel("% variance explained")
    plt.xlabel("Random " + str(i_top+1) + " " + i_gene_name)
    out_file_name = os.path.join(outfolder, 'fract_explained_var'  + outsuffix_2 + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    
    
    yaxmin= min(list(Y[:, i_gene_idx].numpy()) + list(Y_pred[:, i_gene_idx].numpy()) +
                list(Y_pred_c[:, i_gene_idx].numpy()) +
                list(Y_pred_z[:, i_gene_idx].numpy()) +
                list(Y_pred_cz[:, i_gene_idx].numpy()) 
                )
    yaxmax= max(list(Y[:, i_gene_idx].numpy()) + list(Y_pred[:, i_gene_idx].numpy()) +
                list(Y_pred_c[:, i_gene_idx].numpy()) +
                list(Y_pred_z[:, i_gene_idx].numpy()) +
                list(Y_pred_cz[:, i_gene_idx].numpy()) 
                )            

    
            # observed
    plt.scatter(x=mu_z, y=Y[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
    plt.ylim([yaxmin, yaxmax])
    plt.ylabel("Observed values")
    plt.xlabel("z")
    plt.title("Observed data " + i_gene_name + " (random" + str(i_top+1)+')')
    out_file_name = os.path.join(outfolder, 'mapping_z_to_obs' + outsuffix_2 + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    # mapping ND
    plt.scatter(x=mu_z, y=Y_pred[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
    plt.ylim([yaxmin, yaxmax])
    plt.xlabel("z")
    plt.title("ND " + i_gene_name + " (var "+var_type+" random" + str(i_top+1)+')')
    out_file_name = os.path.join(outfolder, 'mapping_z_to_ND_pred' + outsuffix_2 + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    # mapping fz
    plt.scatter(x=mu_z, y=Y_pred_z[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
    plt.ylim([yaxmin, yaxmax])
    plt.xlabel("z")
    plt.title("f(z) " + i_gene_name + " (var "+var_type+" random" + str(i_top+1)+')')
    out_file_name = os.path.join(outfolder, 'mapping_z_to_fz_predz' + outsuffix_2 + '.png')         
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    # mapping fc
    plt.scatter(x=mu_z, y=Y_pred_c[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
    plt.ylim([yaxmin, yaxmax])
    plt.xlabel("z")
    plt.title("f(c) " + i_gene_name + " (var "+var_type+" random" + str(i_top+1)+')')
    out_file_name = os.path.join(outfolder, 'mapping_z_to_fc_predc' + outsuffix_2 + '.png')         
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    # mapping fcz
    plt.scatter(x=mu_z, y=Y_pred_cz[:, i_gene_idx], c=np.array(c_data_bin).reshape(-1))
    plt.ylim([yaxmin, yaxmax])
    plt.xlabel("z")
    plt.title("f(cz) " + i_gene_name + " (var "+var_type+" random" + str(i_top+1)+')')
    out_file_name = os.path.join(outfolder, 'mapping_z_to_fcz_predcz' + outsuffix_2 + '.png')         
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
            

############## show some example of variance explained for some selected genes
# top ranking p-value t test 
tokeep_pos = np.where(df[covarLab].values == "pos") 
mrna_data_pos = mrna_data_scaled[list(tokeep_pos[0]),:]

tokeep_neg = np.where(df[covarLab].values == "neg") 
mrna_data_neg = mrna_data_scaled[list(tokeep_neg[0]),:]

assert mrna_data_neg.shape[0] + mrna_data_pos.shape[0] == mrna_data_filt.shape[0]
assert mrna_data_pos.shape[1] == mrna_data_filt.shape[1]
assert mrna_data_neg.shape[1] == mrna_data_pos.shape[1] 

all_mwu_pvals = []
all_cles = []



for i in range(mrna_data_pos.shape[1]):
    gene_i_pos = mrna_data_pos[:,i]
    gene_i_neg = mrna_data_neg[:,i]
    mwu_test = mwu(x=gene_i_pos, y=gene_i_neg, tail="two-sided")
    p_val = float(mwu_test['p-val'])
    cles = float(mwu_test['CLES'])
    all_mwu_pvals.append(p_val)
    all_cles.append(cles)
    
assert len(all_mwu_pvals) == nfeatures
assert len(all_cles) == nfeatures        

#all_cles_05 = np.vectorize(lambda x: x-0.5)(all_cles)
    
mwu_dt = pd.DataFrame({'gene':gene_names,'MWU_pvals': all_mwu_pvals, 'CLES':all_cles})
mwu_dt['CLES_05'] = mwu_dt['CLES'] - 0.5 
mwu_dt['CLES_05_abs'] = abs(mwu_dt['CLES_05'])
mwu_dt_sorted = mwu_dt.sort_values(by='CLES_05_abs', ascending=False, axis=0)

ntopgenes = 5

for top_g in range(ntopgenes):
    # retrieve matching column
    gene_idx = [i for i,x in enumerate(gene_names) if x == mwu_dt_sorted.gene.values[top_g]]
    assert len(gene_idx) == 1
    gene_idx = gene_idx[0]
    top_gene = gene_names[gene_idx]
    assert top_gene == mwu_dt_sorted.gene.values[top_g]
    # retrieve variance explained
    gene_var_dt = pd.DataFrame(expvar_dt.iloc[gene_idx,:]).T
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=gene_var_dt)
    plt.title("Fraction of variation explained - " + top_gene)
    plt.ylabel("% variance explained")
    plt.xlabel("(ER_Status CLES top gene # " + str(top_g+1)+")")
    out_file_name = os.path.join(outfolder, 'variance_explained_CLES_top' + str(top_g+1)+'_'+ top_gene + outsuffix + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()
    
    
mwu_dt_sorted = mwu_dt.sort_values(by='CLES_05_abs', ascending=True, axis=0)

for top_g in range(ntopgenes):
    # retrieve matching column
    gene_idx = [i for i,x in enumerate(gene_names) if x == mwu_dt_sorted.gene.values[top_g]]
    assert len(gene_idx) == 1
    gene_idx = gene_idx[0]
    top_gene = gene_names[gene_idx]
    assert top_gene == mwu_dt_sorted.gene.values[top_g]
    # retrieve variance explained
    gene_var_dt = pd.DataFrame(expvar_dt.iloc[gene_idx,:]).T
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=gene_var_dt)
    plt.title("Fraction of variation explained - " + top_gene)
    plt.ylabel("% variance explained")
    plt.xlabel("(ER_Status CLES bottom gene # " + str(top_g+1)+")")
    out_file_name = os.path.join(outfolder, 'variance_explained_CLES_bottom' + str(top_g+1)+'_'+ top_gene + outsuffix + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()

rd_idxs = random.sample(range(0, nfeatures), ntopgenes)

r_i=0
rd_idx = rd_idxs[r_i]

for r_i, rd_idx in enumerate(rd_idxs):
    # retrieve matching column
    gene_idx = rd_idx
    rd_gene = gene_names[gene_idx]
    # retrieve variance explained
    gene_var_dt = pd.DataFrame(expvar_dt.iloc[gene_idx,:]).T
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=gene_var_dt)
    plt.title("Fraction of variation explained - " + rd_gene)
    plt.ylabel("% variance explained")
    plt.xlabel("(random gene # " + str(r_i+1)+")")
    out_file_name = os.path.join(outfolder, 'variance_explained_random' + str(r_i + 1)+'_'+ rd_gene + outsuffix + '.png')
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)
    plt.close()

#None available
# annotated_genes = 

#********************
#********************
#********************
print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))
sys.exit(0)

# RERG (Ras-like, oestrogen-regulated, growth-inhibitor) expression in breast cancer: 
# a marker of ER-positive luminal-like subtype
# Within the ERA genes, cyclins A1, A2, B2, E1 and J, cyclin dependent kinase inhibitor 
#2A (CDKN2A) and CDK2 associated protein (CDK2AP1) show higher expression in ER− tumors 
#whereas cyclins D1, G2 and H, CDK7 and cyclin G-associated kinase (GAK) have higher 
#expression in ER+ tumors. Several genes directly involved in DNA replication are more
 #highly expressed in ER− tumors: for example, those encoding proteins in the origin-
 #recognition complex (ORC1L and ORC6L), the minichromosome maintenance proteins MCM2 to MCM7, and CDC45L. 
# https://onlinelibrary.wiley.com/doi/full/10.1002/ijc.22234
 # We infer a robust and reliable signature of 10 genes, which is associated with 
#  ER expression and presumably therapeutically relevant biological processes. 
# Six genes were upregulated in the ER-positive group. Of these, ESR1 and SLC39A6 (LIV-1) 
# are known to be induced by estrogen.21 For FOXA1 and GATA3, each coding for a transcription factor, 
# a coexpression with ESR1 was already reported in the literature.2 Despite the known coexpression, 
# a direct interaction between these transcription factors and ESR1 could not be shown until now.22 
# The 2 genes HADHA and SLC1A4 had not been mentioned in combination with ER status before. 
# Two of the classifier genes (ESR1 and GATA3) were represented by several independent clones (Fig. 2). 
# The gene expression patterns of these clones were very similar, suggesting that different spots
#  for the same gene on the microarrays gave highly concordant results.
# Four genes were found downregulated in the ER-positive group. 
# Three of these coded for immunoglobulin light or heavy chains (IGHG1/IGH@, IGHG3/IGH@, and IGLC2). Expression of immunoglobulins is known to be associated with the infiltration of lymphocytes in the tumor. This, in turn, is inversely correlated to the ER status.2, 4


#annot_genes = ['ESR1','SLC39A6', 'FOXA1','GATA3','HADHA', 'SLC1A4' ,'COL10A1']
#av_annot = [x for x in annot_genes if x in gene_names]
# None...

# 'VEGF' in gene_names
# Out[385]: False
# 
# 'REG' in gene_names
# Out[386]: False

### added MZ
# from     def fraction_of_variance_explained(self, z, c, account_for_noise=False, divide_by_total_var=True):
# z and c are feed forward
expvar_dt = decoder.fraction_of_variance_explained(z=mu_z, c=c)
# collect Var([f_z, f_c, f_int]) together
# and divide by total variance
#f_all_var = torch.cat([f_z_var, f_c_var, f_int_var], dim=0)
expvar_dt.shape
# 1000,3

#out_file = os.path.join(outfolder, "mode.pt")
#torch.save(model)

encoder.mapping
# Out[83]: 
# Sequential(
#   (0): Linear(in_features=6, out_features=32, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=32, out_features=2, bias=True)
# )
# access the weights
encoder.mapping[0].weight.shape
#  torch.Size([32, 6])
decoder.mapping_z[0].weight.shape


# for a given feature, plot value across LS value


# iterate over latent dim to set to 0
# predict
# correlation for each gene with initial value
# scatterplot vs correlation with latent dim


#####################################################
 
 
#col         = 'consumption_energy'#
#conditions  = [ df2[col] >= 400, (df2[col] < 400) & (df2[col]> 200), df2[col] <= 200 ]
#choices     = [ "high", 'medium', 'low' ]
  
# df2["energy_class"] = np.select(conditions, choices, default=np.nan)