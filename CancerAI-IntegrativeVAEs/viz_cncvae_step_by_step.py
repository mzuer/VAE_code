

# python viz_cncvae_step_by_step.py

import datetime
start_time = str(datetime.datetime.now().time())
print('> START: viz_cncvae_step_by_step.py \t' + start_time)

import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import seaborn as sns
import pickle
import numpy as np

import math
from pingouin import mwu


wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','CancerAI-IntegrativeVAEs')
os.chdir(wd)

modelRunFolder = os.path.join('CNCVAE_STEP_BY_STEP')

outfolder = 'VIZ_CNCVAE_STEP_BY_STEP'
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

# scatterplot highest/lowest gene correlation with LD
# retrieve the highest correlation value

correlations_all_df_abs = abs(correlations_all_df)

less_cor = np.unravel_index(correlations_all_df_abs.values.argmin(), correlations_all_df.shape)
pos_max = np.unravel_index(correlations_all_df.values.argmax(), correlations_all_df.shape)
neg_min = np.unravel_index(correlations_all_df.values.argmin(), correlations_all_df.shape)


for i in [less_cor, pos_max, neg_min]:
    i_ld_idx = i[0]  ## 0-based dim
    i_gene_idx = i[1]  ## 0-based dim
    i_gene = correlations_all_df.columns[i_gene_idx]
    gene_expr = df[i_gene].values # do not use index !!! also sample data
    latent_expr = pd.DataFrame(latent_repr).iloc[:,i_ld_idx]
    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(y=gene_expr, x=latent_expr)
    plt.ylabel(i_gene + " expression")
    plt.xlabel("LD " + str(i_ld_idx+1))
    out_file_name = os.path.join(outfolder, i_gene + "_LD " + str(i_ld_idx+1) +"_top_correlations_scatterplot.png")
    plt.savefig(out_file_name, dpi=300) 
    print('... written: ' + out_file_name)


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

# We used the min-max normalisation as unlike other techniques (i.e., Z-score normalisation) it 
# guarantees multi-omics features will have the same scale45. Thus, all the features will have 
# equal importance in the multi-omics analysis

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



#### association with er status
ld_df = pd.DataFrame(emb_train)
ld_df['ER_Status'] = df['ER_Status']
colnames = ld_df.columns.tolist()
colnames[0:64] = [str(x) for x in list(range(1,65))]
ld_df.columns = colnames
sub_dt = ld_df[ld_df.ER_Status != "?"]
    

ndim  = 64
ncol=8
fig, axs = plt.subplots(8,8,figsize = (20,20))

dims_mwu_pvals = [None] * ndim
dims_mwu_cles = [None] * ndim

for i in range(1,65):
    i_row = math.floor((i-1)/ncol)
    i_col = i%ncol - 1
    
    pos_vals = sub_dt[sub_dt['ER_Status'] == "pos"][str(i)]
    neg_vals = sub_dt[sub_dt['ER_Status'] == "neg"][str(i)]

    assert len(pos_vals) + len(neg_vals) == sub_dt.shape[0]
    
    mwu_test = mwu(x=pos_vals, y=neg_vals, tail="two-sided")
    p_val = float(mwu_test['p-val'])
    cles = float(mwu_test['CLES'])
    dims_mwu_pvals[i-1] = p_val
    dims_mwu_cles[i-1] = cles
    
    g = sns.boxplot(x='ER_Status', y=str(i), data=sub_dt, ax=axs[i_row,i_col])
    # MWU test
    g.set(title='LD (pval={:.2e} - CLES={:.2f})'.format(p_val, cles))


out_file_name = os.path.join(outfolder, 'all_boxplots_ERexpr.png')
fig.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)


cles_dt = pd.DataFrame({'LD':range(1,65),'pvals': dims_mwu_pvals, 'CLES': dims_mwu_cles})
cles_dt['CLES_05'] = cles_dt['CLES']-0.5


# NB: do not use next !! is for iterator, in case 'continue'
sig_labels = []
for x in cles_dt['pvals'].values:
    if x < 0.0001:
        sig_labels.append('p<0.0001')
    elif x < 0.001:
        sig_labels.append('p<0.001')
    elif x < 0.01:
        sig_labels.append('p<0.01')
    elif x >= 0.01:
        sig_labels.append('p>=0.01')
    else:
        sys.exit(1)
        
cles_dt['sig'] = sig_labels

plt.figure(figsize=(10,6))
# make barplot
cles_dt_sorted=cles_dt.sort_values(by='CLES_05', ascending=False)
sns.barplot(x='LD', y="CLES_05", data=cles_dt_sorted,
            hue = 'sig',
               order=cles_dt_sorted.sort_values('CLES_05', ascending=False).LD)
# set labels
plt.xlabel("LD", size=15)
plt.ylabel("CLES-0.5", size=15)
plt.title("CLES and sig. LD (MWU)", size=18)
plt.tight_layout()
out_file_name = os.path.join(outfolder, 'all_barplots_CLES.png')
fig.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)


max_cles_LD = cles_dt_sorted['LD'].iloc[0]
min_cles_LD = cles_dt_sorted['LD'].iloc[-1]
        
fig, axs = plt.subplots(1,2,figsize = (10,6))
g = sns.boxplot(x='ER_Status', y=str(max_cles_LD), data=sub_dt,ax=axs[0])
g = sns.boxplot(x='ER_Status', y=str(min_cles_LD), data=sub_dt,ax=axs[1]) 
out_file_name = os.path.join(outfolder, 'min_max_CLES_boxplot.png')
fig.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)


## pairplot for the top 4 (top min and top max)
top2_max = list(cles_dt_sorted.LD[0:2].values.astype(np.str))
top2_min = list(cles_dt_sorted.LD[-2:].values.astype(np.str))

top2_dt = sub_dt.copy()
top2_dt = top2_dt[top2_min+top2_max+['ER_Status']]

sns.pairplot(top2_dt, hue = 'ER_Status')
out_file_name = os.path.join(outfolder, 'top2_CLES_LD_pairplot.png')
fig.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)



#********************
#********************
#********************
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





# ### alternative to compute CLES
# from pingouin import compute_effsize
# compute_effsize(x,y,paired=TRUE,eftype="CLES")
