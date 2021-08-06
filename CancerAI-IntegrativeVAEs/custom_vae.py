#
import datetime
start_time = str(datetime.datetime.now().time())
print('> START: custom_vae.py \t' + start_time)

# python custom_vae.py

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error,binary_crossentropy
from keras.utils.vis_utils import plot_model

import tensorflow as tf

import numpy as np
import os
import sys
import pickle
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import pandas as pd


wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','CancerAI-IntegrativeVAEs')
os.chdir(wd)

#module_path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code'
module_path = os.path.join(wd, 'code')

if module_path not in sys.path:
    sys.path.append(module_path)

from models.common import sse, bce, mmd, sampling, kl_regu
from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding

outfolder = os.path.join('CUSTOM_VAE')
os.makedirs(outfolder, exist_ok=True)


class CNCVAE:
    def __init__(self, args):
        self.args = args
        self.vae = None
        self.encoder = None
        
        self.decoder=None

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
        filename = 'dissect_results/z_mean.sav'
        #z_mean.save(filename)
        pickle.dump(z_mean, open(filename, 'wb'))
    
        z_log_sigma = Dense(self.args.ls, name='z_log_sigma', kernel_initializer='zeros')(x)
        #pickle.dump(self.encoder, open(filename, 'wb'))
        #joblib.dump(self.encoder, filename)

        z = Lambda(sampling, output_shape=(self.args.ls,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        filename = 'dissect_results/encoder.sav'
        #pickle.dump(self.encoder, open(filename, 'wb'))
        #joblib.dump(self.encoder, filename)
        self.encoder.save(filename)
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ls,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        filename = 'dissect_results/x.sav'
        pickle.dump(x, open(filename, 'wb'))
        x = BN()(x)
        
        x=Dropout(self.args.dropout)(x)

        # ------------ Out -----------------------
        
        #if self.args.integration == 'Clin+CNA':
        #    concat_out = Dense(self.args.input_size,activation='sigmoid')(x)
        #else:
        concat_out = Dense(self.args.input_size)(x)
        
        decoder = Model(latent_inputs, concat_out, name='decoder')
        filename = os.path.join(outfolder,'decoder.sav')
        #pickle.dump(decoder, open(filename, 'wb'))
        decoder.save(filename)
        print("... written: " + filename )
        
        decoder.summary()
        
        self.decoder=decoder

        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')
        filename = os.path.join(outfolder, 'vae.sav')
        self.vae.save(filename)
        print("... written: " + filename )
        
        output_model_file = os.path.join(outfolder, 'cncvae_architecture.png')
        plot_model(self.vae, to_file=output_model_file)
        print("... written: " + output_model_file )
        
        output_model_file = os.path.join(outfolder, 'encoder_architecture.png')
        plot_model(self.encoder, to_file=output_model_file)
        print("... written: " + output_model_file )
        
        output_model_file = os.path.join(outfolder, 'decoder_architecture.png')
        plot_model(self.decoder, to_file=output_model_file)
        print("... written: " + output_model_file )

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
        
        outfile = os.path.join(outfolder, 'modelsummary.txt')
        with open(outfile, 'w') as f:
            self.vae.summary(print_fn=lambda x: f.write(x + '\n'))
        print("... written: " + outfile )
            
    def train(self, s_train, s_test):
        train = s_train#np.concatenate((s1_train,s2_train), axis=-1)
        test = s_test#np.concatenate((s1_test,s2_test), axis=-1)
        self.vae.fit(train, epochs=self.args.epochs, batch_size=self.args.bs, shuffle=True, validation_data=(test, None))
        if self.args.save_model:
            #self.vae.save_weights('./models/vae_cncvae.h5')
            self.vae.save_weights(out_model_file)

    def predict(self, s_data):

        return self.encoder.predict(s_data, batch_size=self.args.bs)[0]




parser = argparse.ArgumentParser()
args = parser.parse_args()

###### PARAMETERS TO CHANGE BY THE USER #############

latent_dims = 64
args.ls = latent_dims # latent dimension size
args.ds = 256 # The intermediate dense layers size
args.distance = 'mmd'
args.beta = 1
        

# Exponential Linear Unit (ELU) is a popular activation function that speeds up
#  learning and produces more accurate results. 
# ELU is an activation function based on ReLU that has an extra alpha constant (α) 
# that defines function smoothness when inputs are negative. 
# the negative part is not == 0; the higher alpha, the more distant from the x-axis
args.act = 'elu'
#args.epochs= 150 # init value ow
args.epochs= 150
args.bs= 128  # Batch size

# the batch size limits the number of training sample
# at each epoch will be n_samp/batch_size (rounded)
# e.g. 1980/128 = 15.46 -> 16

args.dropout = 0.2
args.save_model = True

outsuffix = "_" + str(args.epochs) + "epochs_"  + str(args.bs) + "bs"
out_model_file = os.path.join(outfolder, "vae_cncvae" + outsuffix + ".h5")

args.input_size = 1000

###### END #############


cncvae = CNCVAE(args)
cncvae.build_model()

# training data
#df=pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\data\MBdata_33CLINwMiss_1KfGE_1KfCNA.csv') # dataset available in github repo
df=pd.read_csv(os.path.join('data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv')) # dataset available in github repo
mrna_data = df.iloc[:,34:1034].copy().values
mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ (mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)

cncvae.train(mrna_data_scaled, mrna_data_scaled)
emb_train = cncvae.predict(mrna_data_scaled) # this it the latent space representation !

filename = os.path.join(outfolder,'cncvae_vae' + outsuffix + '.sav')
cncvae.vae.save(filename)
print("... written: " + filename )

filename = os.path.join(outfolder,'cncvae_decoder' + outsuffix + '.sav')
cncvae.decoder.save(filename)
print("... written: " + filename )

filename = os.path.join(outfolder,'cncvae_encoder' + outsuffix + '.sav')
cncvae.encoder.save(filename)
print("... written: " + filename )

filename = os.path.join(outfolder,'emb_train' + outsuffix + '.sav')
pickle.dump(emb_train, open(filename, 'wb'))
print("... written: " + filename )

#np.savetxt(r"C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code\results\custom_arch\mRNA_ls64_hs256_mmd_beta1_scaled.csv", emb_train, delimiter = ',')
outfile = os.path.join(outfolder, "mRNA_ls64_hs256_mmd_beta1_scaled" + outsuffix + ".csv")
np.savetxt(outfile, emb_train, delimiter = ',')
print("... written: " + outfile )
###########################################################################################
    
def plot_3plots(data_to_plot, data_with_labels,file_name='', type_ = 'PCA', pca=None):
    
    fig, axs = plt.subplots(1,3,figsize = (15,6))
    palette = 'tab10'
    ### ! depending on the version of matplotlib -> should pass a list to hue !!!!
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        #hue = data_with_labels['ER_Expr'], 
                        hue = list(data_with_labels['ER_Expr']), 
                        ax=axs[0],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        # hue = data_with_labels['Pam50Subtype'], 
                        hue = list(data_with_labels['Pam50Subtype']), 
                        ax=axs[1],linewidth=0, s=15, alpha=0.7, palette = palette)
    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        # hue = data_with_labels['iC10'], 
                        hue = list(data_with_labels['iC10']), 
                        ax=axs[2],linewidth=0, s=15, alpha=0.7, palette = palette)
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
        #out_file_name = os.path.join(outfolder,'downstream_results/{}_{}.png'.format(plot_file_name, type_)) # r -> treated as raw string
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
outfile = os.path.join(outfolder, "latent_repr_tsne")
plot_3plots(data_to_plot=latent_repr_tsne, data_with_labels=df, type_='tSNE', file_name=outfile)
    

# PLOT UMAP for RAW MRNA
mapper = umap.UMAP(n_neighbors=15, n_components=2).fit(mrna_data)
latent_repr_umap = mapper.transform(mrna_data)
outfile = os.path.join(outfolder, "latent_repr_umap_raw")
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

labels = df['Pam50Subtype'].values

lut = dict(zip(set(labels), sns.hls_palette(len(set(labels)))))
col_colors = pd.DataFrame(labels)[0].map(lut)

sns.clustermap(correlations_all_df, col_colors=col_colors)
out_file_name = os.path.join(outfolder, 'correlations_clustermap.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)

sns.clustermap(p_values_all_df)
out_file_name = os.path.join(outfolder, 'pvalues_clustermap.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)


for latent_dim_i in range(latent_dims):
    fig, ax = plt.subplots(figsize=(15,6))
    corrs = correlations_all_df.iloc[latent_dim_i,:]
    corrs.sort_values(ascending=False)[:30].plot.bar(ax=ax)

out_file_name = os.path.join(outfolder, 'correlations_barplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)

for latent_dim_i in range(latent_dims):
    fig, ax = plt.subplots(figsize=(15,6))
    p_values = p_values_all_df.iloc[latent_dim_i,:]
    p_values.sort_values(ascending=True)[:30].plot.bar(ax=ax)
out_file_name = os.path.join(outfolder, 'pvalues_barplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)

    
print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))