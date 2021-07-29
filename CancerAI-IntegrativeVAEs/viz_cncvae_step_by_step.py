
import os
os.chdir('/home/marie/Documents/FREITAS_LAB/VAE_tutos/CancerAI-IntegrativeVAEs/')


import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import seaborn as sns


import pickle
file = open('stepByStep_figures/emb_train.sav', 'rb')
emb_train  = pickle.load(file)


latent_repr = emb_train

# PLOT PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(latent_repr)
latent_repr_pca = pca.transform(latent_repr)
#plot_3plots(data_to_plot=latent_repr_pca, data_with_labels=df, type_='PCA', pca=pca)


data_to_plot=latent_repr_pca
data_with_labels=df

fig, axs = plt.subplots(1,3,figsize = (15,6))
palette = 'tab10'
g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                hue = list(data_with_labels['ER_Expr']), ax=axs[0],linewidth=0, s=15, alpha=0.7, palette = palette)
g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)





    fig, axs = plt.subplots(1,3,figsize = (15,6))
    palette = 'tab10'
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['ER_Expr'], ax=axs[0],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['Pam50Subtype'], ax=axs[1],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['iC10'], ax=axs[2],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    #plt.




outfile = "latent_repr_pca"
plot_3plots(data_to_plot=latent_repr_pca, data_with_labels=df, type_='PCA', pca=pca, file_name=outfile)


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



    fig, axs = plt.subplots(1,3,figsize = (15,6))
    palette = 'tab10'
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['ER_Expr'], ax=axs[0],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['Pam50Subtype'], ax=axs[1],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['iC10'], ax=axs[2],linewidth=0, s=15, alpha=0.7, palette = palette)
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
        out_file_name = r'downstream_results/{}_{}.png'.format(plot_file_name, type_) # r -> treated as raw string
        plt.savefig(out_file_name, dpi=300) 
        print('> saved ' + out_file_name)
    return
    






###########################################################################################
###########################################################################################
###########################################################################################
    
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import seaborn as sns
def plot_3plots(data_to_plot, data_with_labels,file_name='', type_ = 'PCA', pca=None):
    
    fig, axs = plt.subplots(1,3,figsize = (15,6))
    palette = 'tab10'
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['ER_Expr'], ax=axs[0],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['Pam50Subtype'], ax=axs[1],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = data_with_labels['iC10'], ax=axs[2],linewidth=0, s=15, alpha=0.7, palette = palette)
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
        out_file_name = r'downstream_results/{}_{}.png'.format(plot_file_name, type_) # r -> treated as raw string
        plt.savefig(out_file_name, dpi=300) 
        print('> saved ' + out_file_name)
    return
    


latent_repr = emb_train

# PLOT PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(latent_repr)
latent_repr_pca = pca.transform(latent_repr)
#plot_3plots(data_to_plot=latent_repr_pca, data_with_labels=df, type_='PCA', pca=pca)
outfile = "latent_repr_pca"
plot_3plots(data_to_plot=latent_repr_pca, data_with_labels=df, type_='PCA', pca=pca, file_name=outfile)


# PLOT UMAP
data_to_umap = latent_repr
import umap
mapper = umap.UMAP(n_neighbors=15, n_components=2).fit(data_to_umap)
latent_repr_umap = mapper.transform(data_to_umap)
plot_3plots(latent_repr_umap, df, type_='UMAP')
outfile = "latent_repr_umap"
plot_3plots(data_to_plot=latent_repr_umap, data_with_labels=df, type_='UMAP', file_name=outfile)

# PLOT TSNE
from sklearn.manifold import TSNE
latent_repr_tsne = TSNE(n_components=2, perplexity=30 ).fit_transform(latent_repr)
plot_3plots(latent_repr_tsne, df, type_='tSNE')
outfile = "latent_repr_tsne"
plot_3plots(data_to_plot=latent_repr_tsne, data_with_labels=df, type_='tSNE', file_name=outfile)
    

# PLOT UMAP for RAW MRNA
mapper = umap.UMAP(n_neighbors=15, n_components=2).fit(mrna_data)
latent_repr_umap = mapper.transform(mrna_data)
plot_3plots(latent_repr_umap, df, type_='UMAP')

#####################################################################################################################

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
p_values_all = np.array(p_values_all)
p_values_all_df  = pd.DataFrame(p_values_all.T, columns = df.iloc[:,34:1034].columns)


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

import seaborn as sns

labels = df['Pam50Subtype'].values

lut = dict(zip(set(labels), sns.hls_palette(len(set(labels)))))
col_colors = pd.DataFrame(labels)[0].map(lut)


sns.clustermap(correlations_all_df, col_colors=col_colors)

sns.clustermap(p_values_all_df)


for latent_dim_i in range(latent_dims):
    
    fig, ax = plt.subplots(figsize=(15,6))
    
    corrs = correlations_all_df.iloc[latent_dim_i,:]
    
    
    corrs.sort_values(ascending=False)[:30].plot.bar(ax=ax)

for latent_dim_i in range(latent_dims):
    
    fig, ax = plt.subplots(figsize=(15,6))
    
    p_values = p_values_all_df.iloc[latent_dim_i,:]
    
    
    p_values.sort_values(ascending=True)[:30].plot.bar(ax=ax)
    
    
print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))