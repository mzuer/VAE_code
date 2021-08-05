
import sys,os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re

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
latent_dim = 1 # init value: 1

n_iter_integrals = 300 # init value 25000 
logging_freq_integrals = 100 # init value 100
grid_nsteps = 15 # init value 15


outsuffix = "_hd" + str(hidden_dim) + "_nLD" + str(latent_dim) +\
 "_c" + str(covarLab) +  "_" + str(n_iter_integrals) + "_" + str(logging_freq_integrals) +"_"+str(grid_nsteps) 

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

    # output to CPU
    mu_z, sigma_z = mu_z.cpu(), sigma_z.cpu()
    Y_pred = Y_pred.cpu()

assert Y_pred.shape[0] == Y.shape[0]
assert Y_pred.shape[1] == Y.shape[1]
assert Y_pred.shape[0] == nsamp
assert Y_pred.shape[1] == nfeatures
assert mu_z.shape[1] == latent_dim
assert mu_z.shape[0] == nsamp

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
from scipy.stats import spearmanr

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

import seaborn as sns

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
        
        xlab = "LD "+str(i_ld)+1

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
        plt.title(gene_names[i] + " (SCC={:.4f} - pval={:.2e})".format(corr_, p_value))
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
plt.title("Fraction of variation explained")
plt.ylabel("% variance explained")
out_file_name = os.path.join(outfolder, 'sparsity_mask_for_'+sm_col+'_LD'+str(i_ld+1) + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()


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