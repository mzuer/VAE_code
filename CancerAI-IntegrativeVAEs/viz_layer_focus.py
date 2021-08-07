
# python viz_layer_focus.py

import datetime
start_time = str(datetime.datetime.now().time())
print('> START: viz_layer_focus.py \t' + start_time)

import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import seaborn as sns
import pickle
import numpy as np
from scipy.stats import spearmanr
from keras.models import load_model
from pingouin import mwu
import re

wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','CancerAI-IntegrativeVAEs')
os.chdir(wd)

modelRunFolder = os.path.join('CNCVAE_STEP_BY_STEP')

outfolder = 'VIZ_LAYER_FOCUS'
os.makedirs(outfolder, exist_ok=True)

n_ld = 64

n_epochs= 150
batch_size = 128  
outsuffix = "_" + str(n_epochs) + "epochs_" + str(batch_size) + "bs"

z_min = -4.0
z_max = 4.0
grid_nsteps = 10
z_grid = np.linspace(z_min, z_max, num=grid_nsteps)

catvar = 'ER_Status'

### reload data used in first step
vae = load_model(os.path.join(modelRunFolder, 'vae_150epochs_128bs.h5'))
encoder = load_model(os.path.join(modelRunFolder, 'encoder_150epochs_128bs.h5'))
decoder = load_model(os.path.join(modelRunFolder, 'decoder_150epochs_128bs.h5'))

encoder_weights = vae.get_layer('encoder').get_layer('encoding').get_weights()[0]
encoder_weights2 = encoder.get_weights()[0]
assert np.all(encoder_weights == encoder_weights2)

decoder_weights = decoder.get_weights()[6]
decoder_weights2 = vae.get_layer('decoder').get_layer('out').get_weights()[0]
assert np.all(decoder_weights2  == decoder_weights)

file = open(os.path.join(modelRunFolder,'emb_train'+outsuffix+'.sav'), 'rb')
emb_train  = pickle.load(file)

df=pd.read_csv(os.path.join('data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv'))
samp_ids = df['METABRIC_ID']
n_genes = sum(['GE_' in x for x in df.columns])
df.columns = [re.sub('^GE_', '', x) for x in df.columns]

gene_names = df.iloc[:,34:1034].columns
ngenes = len(gene_names)

n_samp = df.shape[0]
assert n_genes == len(gene_names)

mrna_data = df.iloc[:,34:1034].copy().values 
# the values after are CNA, the values before are clinical data

latent_repr = emb_train

assert mrna_data.shape[0] == latent_repr.shape[0]
assert latent_repr.shape[1] == n_ld

########################################################### correlations gene expr with each dim
# for each gene, look how it correlates across samples with each of the latent dimension
print("... start computing correlations gene expr, LD values")
print(str(datetime.datetime.now().time()))
correlations_all=[]
p_values_all=[]
for gene_i in range(mrna_data.shape[1]):
    correlations=[]
    p_values=[]
    for latent_dim_i in range(n_ld):
        
        corr_, p_value = spearmanr(mrna_data[:,gene_i], latent_repr[:,latent_dim_i])
        
        correlations.append(corr_)
        p_values.append(p_value)
        
    correlations_all.append(correlations)
    p_values_all.append(p_values)

correlations_all = np.array(correlations_all)
correlations_all_df = pd.DataFrame(correlations_all.T, columns = gene_names) 
# columns -> retrieve column names from the original data frame
p_values_all = np.array(p_values_all)
p_values_all_df  = pd.DataFrame(p_values_all.T, columns = gene_names)

print("... done")
print(str(datetime.datetime.now().time()))
########################################################### CLES
ld_df = pd.DataFrame(emb_train)
ld_df[catvar] = df[catvar]
colnames = ld_df.columns.tolist()
colnames[0:64] = [str(x) for x in list(range(1,n_ld+1))]
ld_df.columns = colnames
sub_dt = ld_df[ld_df[catvar] != "?"]

dims_mwu_pvals = [None] * n_ld
dims_mwu_cles = [None] * n_ld


print("... start computing CLES of LDs for " + catvar)
print(str(datetime.datetime.now().time()))
for i in range(1,n_ld+1):

    pos_vals = sub_dt[sub_dt[catvar] == "pos"][str(i)]
    neg_vals = sub_dt[sub_dt[catvar] == "neg"][str(i)]

    assert len(pos_vals) + len(neg_vals) == sub_dt.shape[0]
    
    mwu_test = mwu(x=pos_vals, y=neg_vals, tail="two-sided")
    p_val = float(mwu_test['p-val'])
    cles = float(mwu_test['CLES'])
    dims_mwu_pvals[i-1] = p_val
    dims_mwu_cles[i-1] = cles

print("... done")    
print(str(datetime.datetime.now().time()))
    
cles_dt = pd.DataFrame({'LD':range(1,n_ld+1),'pvals': dims_mwu_pvals, 'CLES': dims_mwu_cles})
cles_dt['CLES_05'] = cles_dt['CLES']-0.5

########################################################### find the layer with highest and smallest CLES_05
assert min(cles_dt.LD.values) == 1
min_ld = cles_dt.sort_values(by="CLES_05", ascending=True)['LD'].values[0] 
min_i_ld = min_ld - 1 

max_ld = cles_dt.sort_values(by="CLES_05", ascending=False)['LD'].values[0] 
max_i_ld = max_ld - 1 

###########################################################################################
###########################################################################################
###########################################################################################

###### !!!! warning: the mrna_data contains also the "?" samples

mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ \
(mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)
assert mrna_data_scaled.shape[0] == n_samp
assert mrna_data_scaled.shape[1] == n_genes


# is it the same output to pass full data and then extract or pass one at a time
pred_results_i = encoder.predict(pd.DataFrame(mrna_data_scaled[i,:]).T, batch_size=batch_size)[0]
pred_results_all_i = encoder.predict(mrna_data_scaled, batch_size=batch_size)[0][i,:]
sns.scatterplot(x=pred_results_i[0], y=pred_results_all_i)
# seems ok

pred_rec_i = decoder.predict(pd.DataFrame(emb_train[i,:]).T, batch_size=batch_size)
pred_rec_all_i = decoder.predict(emb_train, batch_size=batch_size)[i,:]
sns.scatterplot(x=pred_rec_i[0], y=pred_rec_all_i)

lds_to_traverse = [min_i_ld, max_i_ld]  # zero-based !
i_samp=0
i_ld = 0
iz=z_grid[0]
# iterate over the LDs
#for i_samp in range(nsamp):

std_outputs = decoder.predict(emb_train, batch_size=batch_size)

all_traversals = dict()

print("... start traversing A")
print(str(datetime.datetime.now().time()))
# for the selected LDs, traverse the grid
for i_ld in lds_to_traverse:
    all_traversals[str(i_ld)] = dict()    
    intact_ld = emb_train.copy()
    # traverse this LD, keep the other LD unchanged
    for iz in z_grid:
        new_ld = intact_ld.copy()
        new_ld[:,i_ld] = iz
        new_outputs = decoder.predict(new_ld, batch_size=batch_size)
        assert new_outputs.shape == std_outputs.shape
        all_traversals[str(i_ld)][str(iz)] = new_outputs
print("... done")    
print(str(datetime.datetime.now().time()))


print("... start plot traversing A")
print(str(datetime.datetime.now().time()))
### for a given LD, imshow of the mrna predicted by varying LD
for i_ld in lds_to_traverse:
    #fig, axs = plt.subplots(1,len(z_grid)+1,figsize = (15,6))
    fig, axs = plt.subplots(1,len(z_grid)+1,figsize = (30,6))
    axs[0].imshow(std_outputs, aspect='auto')
    ###plt.subplot(1, len(z_grid) + 1, 1,figsize=(15,6))
    ###plt.imshow(std_outputs, aspect='auto')
    # traverse this LD, keep the other LD unchanged
    for i_z, iz in enumerate(z_grid):
        ###plt.subplot(1, len(z_grid) + 1, i_z+2)
        ###plt.imshow(all_traversals[str(i_ld)][str(iz)], aspect='auto')
        im=axs[i_z+1].imshow(all_traversals[str(i_ld)][str(iz)], aspect='auto')
    fig.colorbar(im)
    plt.show()
    out_file_name = os.path.join(outfolder, 'LD'+str(i_ld+1) + "_cmp_predicted_latentTrav_heatmaps.png")
    plt.savefig(out_file_name, dpi=300) 
    plt.close()
    print('... written: ' + out_file_name)

print("... done")    
print(str(datetime.datetime.now().time()))


### for a given LD, imshow of the mrna predicted by varying LD - init predicted mrna

print("... start plot traversing B")
print(str(datetime.datetime.now().time()))

for i_ld in lds_to_traverse:
    #fig, axs = plt.subplots(1,len(z_grid)+1,figsize = (15,6))
    fig, axs = plt.subplots(1,len(z_grid)+1,figsize = (30,6))
    axs[0].imshow(std_outputs, aspect='auto')
    ###plt.subplot(1, len(z_grid) + 1, 1,figsize=(15,6))
    ###plt.imshow(std_outputs, aspect='auto')
    # traverse this LD, keep the other LD unchanged
    for i_z, iz in enumerate(z_grid):
        ###plt.subplot(1, len(z_grid) + 1, i_z+2)
        ###plt.imshow(all_traversals[str(i_ld)][str(iz)], aspect='auto')
        mat_diff = all_traversals[str(i_ld)][str(iz)] - std_outputs
        im=axs[i_z+1].imshow(mat_diff, aspect='auto', cmap="RdBu")

    fig.colorbar(im)
    plt.show()
    out_file_name = os.path.join(outfolder, 'LD'+str(i_ld+1) + "_cmp_predicted_latentTrav_diffInit_heatmaps.png")
    plt.savefig(out_file_name, dpi=300) 
    plt.close()
    print('... written: ' + out_file_name)

print("... done")    
print(str(datetime.datetime.now().time()))


##### todo: NOW LOOK AT THE GENES THAT MOST CORRELATED WITH THIS LAYER
##### todo: FOR THIS LAYER IS CORRELATION OF GENE WITH VARYING Z CORRELATES TO GENE EXPR CORRELATION
##### TODO GENE BOXPLOT ACROSS Z FOR ALL SAMPLES

# find the gene that is the most correlated with this LD 

lds_to_traverse2=[int(x) for x in all_traversals.keys()]
assert [str(ld) in list(all_traversals.keys()) for ld in lds_to_traverse]


grid_cols = [str(x) for x in range(grid_nsteps)]
id_cols =  ['i_gene','gene','gene_lab','i_samp', 'sampID', 'i_LD', 'value_grid_SCC_coeff', 'value_grid_SCC_pval']
my_cols = id_cols + grid_cols


# if the aim is to identify a LD of interest based on gene input -> i_ld nested
# if the aim is to identify a gene of interest -> i_gene nested

print("... start selected traversing ")
print(str(datetime.datetime.now().time()))


all_samp_dt = pd.DataFrame(columns = my_cols)
for i_ld in lds_to_traverse:
    leastcorr_gene_i = np.argmin(abs(correlations_all_df.iloc[i_ld,:].values))
    mostnegcorr_gene_i = np.argmin(correlations_all_df.iloc[i_ld,:].values)
    mostposcorr_gene_i = np.argmax(correlations_all_df.iloc[i_ld,:].values)
    
    genes_to_traverse = [leastcorr_gene_i, mostnegcorr_gene_i, mostposcorr_gene_i ]
    genes_to_trav_lab = {leastcorr_gene_i: gene_names[leastcorr_gene_i] + " leastcorr LD" + str(i_ld+1) + " (1b)",
                         mostnegcorr_gene_i:gene_names[mostnegcorr_gene_i] + " mostnegcorr LD" + str(i_ld+1) + " (1b)",
                         mostposcorr_gene_i:gene_names[mostposcorr_gene_i] + " mostposcorr LD" + str(i_ld+1) + " (1b)"
                         }
    for i_gene in genes_to_traverse:
        gene_lab = genes_to_trav_lab[i_gene]
        for i_samp in range(n_samp):
            i_samp_i_gene_ldtravers = []
            for iz in z_grid:
                # retrieve the predicted matrix for the current grid value of this LD
                curr_mat = all_traversals[str(i_ld)][str(iz)] 
                assert curr_mat.shape[1] == ngenes
                i_samp_i_gene_ldtravers.append(curr_mat[i_samp,i_gene])
            assert len(i_samp_i_gene_ldtravers) == grid_nsteps
            # correlation of the predicted expression with the grid value
            corr, p_val = spearmanr(i_samp_i_gene_ldtravers, z_grid)
            sns.scatterplot(x=z_grid, y =i_samp_i_gene_ldtravers)
            lt_dt = pd.DataFrame(i_samp_i_gene_ldtravers).T
            lt_dt.columns = grid_cols
            id_dt = pd.DataFrame([i_gene, gene_names[i_gene],gene_lab,i_samp,samp_ids[i_samp],i_ld, corr, p_val]).T
            id_dt.columns = id_cols
            ig_is_dt= pd.concat([id_dt, lt_dt], axis=1)
            all_samp_dt = pd.concat([all_samp_dt, ig_is_dt], axis=0)
        
        # for a given gene, a given LD, boxplot of the predicted values along traversals
        sub_dt = all_samp_dt[(all_samp_dt['i_gene'] == i_gene) & (all_samp_dt['i_LD'] == i_ld)].copy()
        assert np.all(sub_dt ['i_gene'].values == i_gene)
        assert np.all(sub_dt ['i_LD'].values == i_ld)
        assert sub_dt.shape[0] > 0
        
        nsamp_trav = len(set(list(sub_dt['i_samp'])))
        assert sub_dt.shape[0] == nsamp_trav
        # do boxplot across LD variation
        box_data = sub_dt.copy()
        box_data = box_data[grid_cols]
        sns.boxplot(data=box_data)
        x=plt.xticks(list(range(len(z_grid))), list(z_grid.round(2)))
        x=plt.ylabel("Predicted gene expr.", size=12)
        x=plt.xlabel("Values of LD " + str(i_ld+1), size=12)
        x=plt.suptitle("Effect of LD variation on pred. expr. - " + gene_names[i_gene], size=14)
        x=plt.title("(# samp = "+str(nsamp_trav)+")", size=10)
                
        out_file_name = os.path.join(outfolder, 'pred_gene_expr_latentTrav_' + gene_lab.replace(' ','') + '.png')
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
        plt.close()

        
print("... done")    
print(str(datetime.datetime.now().time()))

all_samp_dt_A = all_samp_dt.copy()
filename = os.path.join(outfolder, 'all_samp_dt_A.sav')
pickle.dump(all_samp_dt_A, open(filename, 'wb'))
print("... written: " + filename )

        
##### compare with the correlations obtained 

print("... start traverse corr ")
print(str(datetime.datetime.now().time()))


lds_to_traverse=lds_to_traverse
all_samp_dt = pd.DataFrame(columns = my_cols)
nsamp_trav = n_samp
for i_ld in lds_to_traverse:
    for i_gene in range(len(gene_names)):
        gene_lab = gene_names[i_gene]
        for i_samp in range(nsamp_trav):
            i_samp_i_gene_ldtravers = []
            for iz in z_grid:
                # retrieve the predicted matrix for the current grid value of this LD
                curr_mat = all_traversals[str(i_ld)][str(iz)] 
                assert curr_mat.shape[1] == ngenes
                i_samp_i_gene_ldtravers.append(curr_mat[i_samp,i_gene])
            assert len(i_samp_i_gene_ldtravers) == grid_nsteps
            # correlation of the predicted expression with the grid value
            corr, p_val = spearmanr(i_samp_i_gene_ldtravers, z_grid)
            lt_dt = pd.DataFrame(i_samp_i_gene_ldtravers).T
            lt_dt.columns = grid_cols
            id_dt = pd.DataFrame([i_gene, gene_names[i_gene],gene_lab,i_samp,samp_ids[i_samp],i_ld, corr, p_val]).T
            id_dt.columns = id_cols
            ig_is_dt= pd.concat([id_dt, lt_dt], axis=1)
            all_samp_dt = pd.concat([all_samp_dt, ig_is_dt], axis=0)
        
print("... done")    
print(str(datetime.datetime.now().time()))

filename = os.path.join(outfolder, 'all_samp_dt_B.sav')
pickle.dump(all_samp_dt, open(filename, 'wb'))
print("... written: " + filename )


# for each gene, for each LD, take the mean of the corr of expression with the traversal
gene_mean_corr_trav = all_samp_dt.groupby(['i_gene', 'i_LD']).value_grid_SCC_coeff.apply(np.mean).reset_index().copy()

corr_all_dt = pd.DataFrame(correlations_all)
corr_all_dt['i_gene'] = range(corr_all_dt.shape[0])
corr_all_m = pd.melt(corr_all_dt,id_vars=['i_gene'],var_name='i_LD', value_name='corr')

cmp_dt = gene_mean_corr_trav.merge(corr_all_m, 'inner', on=['i_gene', 'i_LD'])
sns.scatterplot(x=cmp_dt['corr'], y =cmp_dt['value_grid_SCC_coeff'])
x=plt.ylabel("Mean grid SCC coeff.", size=12)
x=plt.xlabel("Mean corr expr. and LD values")
x=plt.title("(# samp="+str(nsamp_trav)+"-# genes=" + str(len(gene_names)) + "-# LDs="+str(len(lds_to_traverse)))
        
out_file_name = os.path.join(outfolder, 'corr_SCC_LD_latentTrav_all.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()


#################### END
print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))

sys.exit(0)
##### compare with the correlations obtained 


print("... start compute corr ")
print(str(datetime.datetime.now().time()))

correlations_all=[]
p_values_all=[]
for gene_i in range(mrna_data.shape[1]):
    correlations=[]
    p_values=[]
    for latent_dim_i in range(n_ld):
        
        corr_, p_value = spearmanr(mrna_data[:,gene_i], latent_repr[:,latent_dim_i])
        
        correlations.append(corr_)
        p_values.append(p_value)
        
    correlations_all.append(correlations)
    p_values_all.append(p_values)

correlations_all = np.array(correlations_all)
correlations_all_df = pd.DataFrame(correlations_all.T, columns = df.iloc[:,34:1034].columns) 
# columns -> retrieve column names from the original data frame
p_values_all = np.array(p_values_all)
p_values_all_df  = pd.DataFrame(p_values_all.T, columns = df.iloc[:,34:1034].columns)

print("... done")    
print(str(datetime.datetime.now().time()))

