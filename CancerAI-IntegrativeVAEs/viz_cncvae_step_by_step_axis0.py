

# python viz_cncvae_step_by_step_axis0.py

import datetime
start_time = str(datetime.datetime.now().time())
print('> START: cncvae_step_by_step_axis0.py \t' + start_time)



import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import seaborn as sns
import pickle
import numpy as np

wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','CancerAI-IntegrativeVAEs')
os.chdir(wd)

modelRunFolder = os.path.join('CNCVAE_STEP_BY_STEP_AXIS0')

outfolder = 'VIZ_CNCVAE_STEP_BY_STEP_AXIS0'
os.makedirs(outfolder, exist_ok=True)

latent_dims = 64

n_epochs= 150
batch_size = 128  
outsuffix = "_" + str(n_epochs) + "epochs_" + str(batch_size) + "bs"


### reload data used in first step
file = open(os.path.join(modelRunFolder,'emb_train'+outsuffix+'.sav'), 'rb')
emb_train  = pickle.load(file)

df=pd.read_csv(os.path.join('data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv'))

n_samp = df.shape[0]
n_genes = sum(['GE_' in x for x in df.columns])

mrna_data = df.iloc[:,34:1034].copy().values 
# the values after are CNA, the values before are clinical data



# plt.subplots() is a function that returns a tuple containing a figure and axes object(s). 
# Thus when using fig, ax = plt.subplots() you unpack this tuple into the variables fig and ax. 
# Having fig is useful if you want to change figure-level attributes or save the figure as an
# image file later (e.g. with fig.savefig('yourfilename.png')). You certainly don't have to
 #use the returned figure object but many people do use it later so it's common to see.
# Also, all axes objects (the objects that have plotting methods), have a parent figure
# object anyway, thus:
# fig, ax = plt.subplots()
# is more concise than this:
# fig = plt.figure()
# ax = fig.add_subplot(111)




###########################################################################################
###########################################################################################
###########################################################################################
    
def plot_3plots(data_to_plot, data_with_labels,file_name='', type_ = 'PCA', pca=None):
    
    fig, axs = plt.subplots(1,3,figsize = (15,6))
    palette = 'tab10'
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = list(data_with_labels['ER_Expr']), ax=axs[0],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = list(data_with_labels['Pam50Subtype']), ax=axs[1],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = list(data_with_labels['iC10']), ax=axs[2],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    #plt.
    # ax[0].plot(latent_repr_pca[:,0], latent_repr_pca[:,1], '.')
    # ax[0].plot(latent_repr_pca[:,0], latent_repr_pca[:,1], '.')
    
    for ax in axs:
        ax.set_xlabel('{} 1'.format(type_))
        ax.set_ylabel('{} 2'.format(type_))
    
    if type_ =='PCA':
        fig.suptitle('{}\nPCA - explained variance ratio: {:.2f}'.format(file_name,pca.explained_variance_ratio_.sum()), x=0.5, y=0.99)
    else:
        fig.suptitle('{}\n{}'.format(file_name,type_), x=0.5, y=0.99)
        
    plt.tight_layout()
    
    if file_name != '':
        plot_file_name = str.replace(file_name, '\\','_').split('.')[0]
        out_file_name = os.path.join('{}_{}.png'.format(plot_file_name, type_)) # r -> treated as raw string
        plt.savefig(out_file_name, dpi=300) 
        print('... written: ' + out_file_name)
    return
    


latent_repr = emb_train

# PLOT PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(latent_repr)
latent_repr_pca = pca.transform(latent_repr)
#plot_3plots(data_to_plot=latent_repr_pca, data_with_labels=df, type_='PCA', pca=pca)
outfile = os.path.join(outfolder, "latent_repr_pca")
plot_3plots(data_to_plot=latent_repr_pca, data_with_labels=df, type_='PCA', pca=pca, file_name=outfile)


# PLOT UMAP
data_to_umap = latent_repr
import umap
mapper = umap.UMAP(n_neighbors=15, n_components=2).fit(data_to_umap)
latent_repr_umap = mapper.transform(data_to_umap)
plot_3plots(latent_repr_umap, df, type_='UMAP')
outfile = os.path.join(outfolder, "latent_repr_umap")
plot_3plots(data_to_plot=latent_repr_umap, data_with_labels=df, type_='UMAP', file_name=outfile)

# PLOT TSNE
from sklearn.manifold import TSNE
latent_repr_tsne = TSNE(n_components=2, perplexity=30 ).fit_transform(latent_repr)
plot_3plots(latent_repr_tsne, df, type_='tSNE')
outfile = os.path.join(outfolder, "latent_repr_tSNE")
plot_3plots(data_to_plot=latent_repr_tsne, data_with_labels=df, type_='tSNE', file_name=outfile)
    

# PLOT UMAP for RAW MRNA
mapper = umap.UMAP(n_neighbors=15, n_components=2).fit(mrna_data)
latent_repr_umap = mapper.transform(mrna_data)
outfile = os.path.join(outfolder, "latent_repr_umap_raw")
plot_3plots(latent_repr_umap, df, type_='UMAP', file_name=outfile)

#####################################################################################################################

# for each gene, look how it correlates across samples with each of the latent dimension

from scipy.stats import spearmanr
correlations_all=[]
p_values_all=[]
for gene_i in range(mrna_data.shape[1]):
    correlations=[]
    p_values=[]
    for latent_dim_i in range(latent_dims):
        
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

# for each gene: correlation with each of the latent dim -> 1000 x 64 (iterate over genes, iterate over LD)
# _all.shape : 1000, 64
#_all_df.shape: 64 x 1000


# max(mrna_data_scaled[:,1])
# Out[22]: 1.0
# min(mrna_data_scaled[:,1])
# Out[23]: 0.4649525747266828
# min(mrna_data_scaled[1,:])
# Out[24]: 0.0
# max(mrna_data_scaled[1,:])
# Out[25]: 1.0
# => the data have been scaled by sample !

# Hira et al. ovarian cancer
# We used the min-max normalisation as unlike other techniques (i.e., Z-score normalisation) 
# it guarantees multi-omics features will have the same scale45. Thus, all the features will
#  have equal importance in the multi-omics analysis

# VAE Cancer multi-omics integration
# 1000 features of normalized gene expression numerical data, scaled to [0,1]

# Franco et al. 2021
# we scaled each data using the following equation.
# Xn=Xi−xminxmax−xmin
# (1)
# where Xi is the data instance while xmax and xmin are the minimum and 
# maximum absolute value of feature X respectively, and Xn is the feature after normalization. 


labels = df['Pam50Subtype'].values

# seaborn.hls_palette(n_colors=6, h=0.01, l=0.6, s=0.65, as_cmap=False)
#     Get a set of evenly spaced colors in HLS hue space.
#     h, l, and s should be between 0 and 1
#            n= number of colors in the palette
#             h=first hue
#             l=lightness
#             s=saturation
lut = dict(zip(set(labels), sns.hls_palette(len(set(labels)))))
col_colors = pd.DataFrame(labels)[0].map(lut)


# cluster samples by correlation of gene
sns.clustermap(correlations_all_df, col_colors=col_colors)
out_file_name = os.path.join(outfolder, 'correlations_clustermap.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)




##### a way to get the hierarchy:
from scipy.spatial import distance
from scipy.cluster import hierarchy

correlations = correlations_all_df
correlations_array = np.asarray(correlations_all_df)

row_linkage = hierarchy.linkage(
    distance.pdist(correlations_array), method='average')

col_linkage = hierarchy.linkage(
    distance.pdist(correlations_array.T), method='average')

sns.clustermap(correlations_all_df, row_linkage=row_linkage, 
               col_linkage=col_linkage, col_colors=col_colors)
               
# cluster samples by correlation pvals of gene
sns.clustermap(p_values_all_df)
out_file_name = os.path.join(outfolder, 'pvals_clustermap.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)

# for each of the LD, barplot of the 30 most correlated genes
for latent_dim_i in range(latent_dims):
    fig, ax = plt.subplots(figsize=(15,6))
    corrs = correlations_all_df.iloc[latent_dim_i,:]
    corrs.sort_values(ascending=False)[:30].plot.bar(ax=ax)
out_file_name = os.path.join(outfolder, 'correlations_barplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)


    
# for each of the LD, barplot of the 30 highest pvalues genes
for latent_dim_i in range(latent_dims):
    fig, ax = plt.subplots(figsize=(15,6))
    p_values = p_values_all_df.iloc[latent_dim_i,:]
    p_values.sort_values(ascending=True)[:30].plot.bar(ax=ax)
out_file_name = os.path.join(outfolder, 'pvalues_barplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)

print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))
sys.exit(0)

# do example boxplot lat dim 46 et 8 
# 1980,64
ld_df = pd.DataFrame(emb_train)
ld_df['Pam50Subtype'] = df['Pam50Subtype']
ld_df['ER_Status'] = df['ER_Status']

colnames = ld_df.columns.tolist()
colnames[0:64] = [str(x) for x in list(range(1,65))]
ld_df.columns = colnames
sns.boxplot(x='ER_Status', y="8", data=ld_df)
    
ld_df[~ld_df.ER_Status.eq("?")] # 1938=66
ld_df[ld_df.ER_Status.ne("?")]
ld_df[ld_df['ER_Status'].isin(list("?")) == False] # can handle multiple values then
ld_df[ld_df.ER_Status != "?"]

sub_dt = ld_df[ld_df.ER_Status != "?"]
sns.boxplot(x='ER_Status', y="64", data=sub_dt)
    


### to compute CLES
from pingouin import compute_effsize

compute_effsize(x,y,paired=TRUE,eftype="CLES")