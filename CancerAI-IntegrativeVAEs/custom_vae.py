#
import datetime
print('> START: ' + str(datetime.datetime.now().time()))


from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from tensorflow.keras.models import Model

import tensorflow as tf
import numpy as np

import os
import sys
#module_path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code'
module_path = r'/home/marie/Documents/FREITAS_LAB/VAE_tutos/CancerAI-IntegrativeVAEs/code'

os.chdir('/home/marie/Documents/FREITAS_LAB/VAE_tutos/CancerAI-IntegrativeVAEs/')

if module_path not in sys.path:
    sys.path.append(module_path)

from models.common import sse, bce, mmd, sampling, kl_regu
from tensorflow.keras.losses import mean_squared_error,binary_crossentropy
import numpy as np



from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding



class CNCVAE:
    def __init__(self, args):
        self.args = args
        self.vae = None
        self.encoder = None

    def build_model(self):
        np.random.seed(42)
        # tf.random.set_random_seed(42)   # moved to (not able to install tensorflow 1.11)
        tf.random.set_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        inputs = Input(shape=(self.args.input_size,), name='concat_input')
        #inputs = [concat_inputs]

        # ------------ Encoding Layer -----------------
        x = Dense(self.args.ds, activation=self.args.act)(inputs)
        x = BN()(x)      

        # ------------ Embedding Layer --------------
        z_mean = Dense(self.args.ls, name='z_mean')(x)
        z_log_sigma = Dense(self.args.ls, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(sampling, output_shape=(self.args.ls,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ls,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)
        
        x=Dropout(self.args.dropout)(x)

        # ------------ Out -----------------------
        
        #if self.args.integration == 'Clin+CNA':
        #    concat_out = Dense(self.args.input_size,activation='sigmoid')(x)
        #else:
        concat_out = Dense(self.args.input_size)(x)
        
        decoder = Model(latent_inputs, concat_out, name='decoder')
        decoder.summary()

        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        # Define the loss
        if self.args.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.args.bs, self.args.ls]))
            distance = mmd(true_samples, z)
        if self.args.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)


        #if self.args.integration == 'Clin+CNA':
        #    reconstruction_loss = binary_crossentropy(inputs, outputs)
        #else:
        reconstruction_loss = mean_squared_error(inputs, outputs)
        vae_loss = K.mean(reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
        self.vae.compile(optimizer=adam)
        self.vae.summary()

    def train(self, s_train, s_test):
        train = s_train#np.concatenate((s1_train,s2_train), axis=-1)
        test = s_test#np.concatenate((s1_test,s2_test), axis=-1)
        self.vae.fit(train, epochs=self.args.epochs, batch_size=self.args.bs, shuffle=True, validation_data=(test, None))
        if self.args.save_model:
            #self.vae.save_weights('./models/vae_cncvae.h5')
            self.vae.save_weights(out_model_file)

    def predict(self, s_data):

        return self.encoder.predict(s_data, batch_size=self.args.bs)[0]





import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

###### PARAMETERS TO CHANGE BY THE USER #############

latent_dims = 64
args.ls = latent_dims # latent dimension size
args.ds = 256 # The intermediate dense layers size
args.distance = 'mmd'
args.beta = 1
        
args.act = 'elu'
#args.epochs= 150 # init value ow
args.epochs= 150
args.bs= 128  # Batch size
args.dropout = 0.2
args.save_model = True

out_model_file = "results/custom_vae/vae_cncvae.h5"

args.input_size = 1000

###### END #############


cncvae = CNCVAE(args)
cncvae.build_model()

# training data
import pandas as pd
#df=pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\data\MBdata_33CLINwMiss_1KfGE_1KfCNA.csv') # dataset available in github repo
df=pd.read_csv(r'data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv') # dataset available in github repo
mrna_data = df.iloc[:,34:1034].copy().values
mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ (mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)

cncvae.train(mrna_data_scaled, mrna_data_scaled)
emb_train = cncvae.predict(mrna_data_scaled) # this it the latent space representation !


#np.savetxt(r"C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code\results\custom_arch\mRNA_ls64_hs256_mmd_beta1_scaled.csv", emb_train, delimiter = ',')
np.savetxt(r"results/custom_vae/mRNA_ls64_hs256_mmd_beta1_scaled.csv", emb_train, delimiter = ',')

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
    #plot_file_name = str.replace(file_name, '\\','_').split('.')[0]
    #plt.savefig(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\plots\{}_{}.png'.format(plot_file_name, type_), dpi=300)
    
    return
    


latent_repr = emb_train

# PLOT PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(latent_repr)
latent_repr_pca = pca.transform(latent_repr)
plot_3plots(latent_repr_pca, df, type_='PCA', pca=pca)


# PLOT UMAP
data_to_umap = latent_repr
import umap
mapper = umap.UMAP(n_neighbors=15, n_components=2).fit(data_to_umap)
latent_repr_umap = mapper.transform(data_to_umap)
plot_3plots(latent_repr_umap, df, type_='UMAP')

# PLOT TSNE
from sklearn.manifold import TSNE
latent_repr_tsne = TSNE(n_components=2, perplexity=30 ).fit_transform(latent_repr)
plot_3plots(latent_repr_tsne, df, type_='tSNE')
    

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
    
    
print('***** DONE ' + str(datetime.datetime.now().time()))