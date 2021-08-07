
# python cncvae_step_by_step.py

#### load modules



import datetime
start_time = str(datetime.datetime.now().time())
print('> START: cncvae_step_by_step.py \t' + start_time)

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error,binary_crossentropy
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot

from scipy.stats import spearmanr

import tensorflow as tf

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math
import umap
import re

wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','CancerAI-IntegrativeVAEs')
os.chdir(wd)

#module_path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code'
module_path = os.path.join(wd, 'code')


if module_path not in sys.path:
    sys.path.append(module_path)

from models.common import sse, bce, mmd, sampling, kl_regu
from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding


#### set hard-coded params
latent_dims = 64
denselayer_size = 256 # The intermediate dense layers size
loss_distance = 'mmd'
loss_beta = 1

activ_fct = 'elu'
#args.epochs= 150 # init value ow
n_epochs= 150
batch_size = 128  
dropout_ratio = 0.2

outfolder = os.path.join('CNCVAE_STEP_BY_STEP')
os.makedirs(outfolder, exist_ok=True)

outsuffix = "_" + str(n_epochs) + "epochs_" + str(batch_size) + "bs"

###### load data

# training data
df=pd.read_csv(os.path.join('data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv'))

n_samp = df.shape[0]
n_genes = sum(['GE_' in x for x in df.columns])

mrna_data = df.iloc[:,34:1034].copy().values 
gene_names = [re.sub('GE_', '', x) for x in df.columns[34:1034]]
samp_ids = df['METABRIC_ID']
# the values after are CNA, the values before are clinical data
# copy() for deep copy
# values to convert to multidim array
mrna_data2= df.filter(regex='GE_').copy().values
assert mrna_data2.shape == mrna_data.shape
# 1980 x 2034
# with sample in rows, genes as columns

mrna_data.min(axis=1).shape
# (1980,)
mrna_data.min(axis=1).reshape(-1,1).shape
# (1980,1)

mrna_data.min(axis=0).shape
# 1000

# see notes in viz
# rescale to 0-1 range
# x-min/max-min

toy_df = np.array([[1,1,1,1],
                   [2,2,2,2],
                   [3,3,3,3]])
toy_df.shape
# 3,4
toy_df.min(axis=1)  # works over the cols shape[1], dim = shape[0] 
# 1,2,3,
toy_df.min(axis=0) # works over the cols shape[0], dim = shape[1]
# [1,1,1,1]

# axis=1 row-wise [for each row], along the columns
# axis=0 column-wise [for each column], along the rows
# reshape(-1) reshapes as line vector
# reshape(-1,1) reshapes as column vector
mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ \
(mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)
# len(mrna_data.min(axis=1))  Out[218]: 1980  => took min and max of samples
 
# After missing-data removal, the input data sets consisted of 1000 features of
#  normalized gene expression numerical data, scaled to [0,1], and 1000 features 
# of copy number categorical data.

# We used the min-max normalisation as unlike other techniques (i.e., Z-score normalisation) it 
# guarantees multi-omics features will have the same scale45. Thus, all the features will have 
# equal importance in the multi-omics analysis

assert mrna_data_scaled.shape[0] == n_samp
assert mrna_data_scaled.shape[1] == n_genes

# mrna_data.min(axis=1).shape
# (1980,)
assert mrna_data.min(axis=1).shape[0] == n_samp
assert np.all((mrna_data_scaled.min(axis=1) == 0))
assert np.all((mrna_data_scaled.max(axis=1) == 1))

### NORMALIZATION HAS BEEN DONE BY SAMPLES !!!!

# mrna_data.min(axis=0).shape
# (1000,)
assert mrna_data.min(axis=0).shape[0] == n_genes
#assert np.all((mrna_data_scaled.min(axis=0) == 0))  # not True !!!
#assert np.all((mrna_data_scaled.max(axis=0) == 1)) # not True !!!

input_size = 1000
assert input_size == n_genes
assert input_size == mrna_data_scaled.shape[1]

np.random.seed(42)
tf.random.set_seed(42)


# Build the encoder network
# ------------ Input -----------------
inputs = Input(shape=(input_size,), name='concat_input')
#inputs = [concat_inputs]

# ------------ Encoding Layer -----------------
x = Dense(denselayer_size, activation=activ_fct, name="encoding")(inputs)
x = BN()(x)      # batch normalization


# ------------ Embedding Layer --------------
z_mean = Dense(latent_dims, name='z_mean')(x)
z_log_sigma = Dense(latent_dims, name='z_log_sigma', kernel_initializer='zeros')(x)
# The Lambda layer exists so that arbitrary expressions can be used as a Layer 
# when constructing Sequential and Functional API models.
z = Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_sigma])

# Model(inputs, outputs, name)
encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
encoder2 = Model(inputs, [z,z_mean, z_log_sigma], name='encoder')

# where you start from Input, you chain layer calls to specify the model's forward pass, 
# and finally you create your model from inputs and outputs:
# inputs = tf.keras.Input(shape=(3,))
# x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
# outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)


# Build the decoder network
# ------------ Dense out -----------------
latent_inputs = Input(shape=(latent_dims,), name='z_sampling')
x = latent_inputs
x = Dense(denselayer_size, activation=activ_fct, name="decoding")(x)
x = BN()(x)

x=Dropout(dropout_ratio)(x)

# ------------ Out -----------------------

#if self.args.integration == 'Clin+CNA':
#    concat_out = Dense(self.args.input_size,activation='sigmoid')(x)
#else:
out_layer = Dense(input_size, name="out")
concat_out = out_layer(x)
# concat_out = Dense(input_size, name="out")(x) => this will not work if I wont to retrieve the weights
# ===> There's a difference between the layer (Dense(n)) and the output tensor you get 
# when applying this layer to some input tensor (Dense(n)(input)). 
# You need to store the layer in a variable, not just the output tensor:

decoder = Model(latent_inputs, concat_out, name='decoder')
decoder.summary()


outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# should be done before compiling
output_model_file = os.path.join(outfolder, 'cncvae_architecture.png')
plot_model(vae, to_file=output_model_file)
print("... written: " + output_model_file )

output_model_file = os.path.join(outfolder, 'encoder_architecture.png')
plot_model(encoder, to_file=output_model_file)
print("... written: " + output_model_file )

output_model_file = os.path.join(outfolder, 'decoder_architecture.png')
plot_model(decoder, to_file=output_model_file)
print("... written: " + output_model_file )

# Define the loss
if loss_distance == "mmd":
    true_samples = K.random_normal(K.stack([batch_size, latent_dims]))
    distance = mmd(true_samples, z)
if loss_distance == "kl":
    distance = kl_regu(z_mean,z_log_sigma)


#if self.args.integration == 'Clin+CNA':
#    reconstruction_loss = binary_crossentropy(inputs, outputs)
#else:
reconstruction_loss = mean_squared_error(inputs, outputs)
vae_loss = K.mean(reconstruction_loss + loss_beta * distance)
vae.add_loss(vae_loss)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
vae.compile(optimizer=adam)  # can add metrics argument .e.g metrics=['accuracy'¨]
vae.summary()

outfile = os.path.join(outfolder,'encoder_modelsummary.txt')
with open(outfile, 'w') as f:
    encoder.summary(print_fn=lambda x: f.write(x + '\n'))
print("... written: " + outfile)
outfile = os.path.join(outfolder,'decoder_modelsummary.txt')
with open(outfile, 'w') as f:
    decoder.summary(print_fn=lambda x: f.write(x + '\n'))
print("... written: " + outfile)
outfile = os.path.join(outfolder, 'modelsummary.txt')
with open(outfile, 'w') as f:
    vae.summary(print_fn=lambda x: f.write(x + '\n'))
print("... written: " + outfile)

s_train = mrna_data_scaled
s_test = mrna_data_scaled

ds_train = s_train#np.concatenate((s1_train,s2_train), axis=-1)
ds_test = s_test#np.concatenate((s1_test,s2_test), axis=-1)

# from cncvae.train():
history = vae.fit(ds_train, epochs=n_epochs, batch_size=batch_size, 
                  shuffle=True, validation_data=(ds_test, None))
filename = os.path.join(outfolder, 'vae_history'+ outsuffix +'.sav')
pickle.dump(history.history, open(filename, 'wb'))
print("... written: " + filename )

out_weights_file = os.path.join(outfolder, 'vae_weights'+ outsuffix +'.h5')
vae.save_weights(out_weights_file)

out_model_file = os.path.join(outfolder, 'vae'+ outsuffix +'.h5')
vae.save(out_model_file)

# from cncvae.predict():
#encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
pred_results = encoder.predict(mrna_data_scaled, batch_size=batch_size)
emb_train = pred_results[0]

pred_results2 = encoder2.predict(mrna_data_scaled, batch_size=batch_size)
emb_train2 = pred_results2[2]

### the subsection from encoder predict comes from here: 
#encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
#encoder2 = Model(inputs, [z,z_mean, z_log_sigma], name='encoder')
assert np.array_equal(pred_results[0], pred_results2[1])
assert np.array_equal(pred_results[1], pred_results2[2])
# assert np.array_equal(pred_results[2], pred_results2[0]) not true because random sample step

filename = os.path.join(outfolder, 'emb_train'+ outsuffix +'.sav')
pickle.dump(emb_train, open(filename, 'wb'))
print("... written: " + filename )

#################### plot the training performance
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,n_epochs+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()

filename = os.path.join(outfolder, 'train_valid_loss'+ outsuffix +'.png')
plt.savefig(filename, dpi=300) 
print('... written: ' + filename)
plt.close()

#################### look at the weights

#vae_weights = model.load(vae_weights_150epochs_128bs.h5)
# from keras.models import load_model
# vae_loaded = load_model('CNCVAE_STEP_BY_STEP/vae_150epochs_128bs.h5')

vae_loaded = vae

all_weights = dict()
all_bias = dict()

i=0
for layer in vae_loaded.layers:
  print("> Layer # " + str(i))
  i+=1
  print(layer.name)
  #print("Weights")
  print(len(layer.get_weights()))
  if len(layer.get_weights()) > 0: 
      print("...Shape weights: ",layer.get_weights()[0].shape)
      all_weights[layer.name] = layer.get_weights()[0]
      print("...Shape biases: ",layer.get_weights()[1].shape)
      all_bias[layer.name] = layer.get_weights()[1]
  else:
      print("No weights/biases")

# vae_loaded.layers[0].get_weights()[0]
# first_layer_weights = model.layers[0].get_weights()[0]
# first_layer_biases  = model.layers[0].get_weights()[1]
# second_layer_weights = model.layers[1].get_weights()[0]
# second_layer_biases  = model.layers[1].get_weights()[1]
# 
      
all_weights.keys()

comparison =  all_weights['encoder'] == all_weights['encoding']
assert comparison.all()      


comparison =  all_weights['encoder'] == encoder.get_weights()[0]
assert comparison.all()      

all_weights['encoder'].shape
# 1000 x 256

all_weights['encoding'].shape
# 1000 x 256

all_weights['decoder'].shape # ?????????????????????
# 64 x 256

encoder_weights = all_weights['encoding'].flatten()

sns.distplot(encoder_weights, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Dist. encoder weights')
plt.xlabel('Weights')
plt.ylabel('Features (genes)')

out_file_name = os.path.join(outfolder, 'encoder_weights_distplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

# find the gene that has the max activation
enc_weights_dt = pd.DataFrame(all_weights['encoding'])
maxvals = enc_weights_dt.apply(max, axis=1)
np.argmax(maxvals)

df.columns[34:1034][np.argmax(maxvals)]
# The transcription factor DEC1 (stra13, SHARP2) is associated with the 
# hypoxic response and high tumour grade in human breast cancers

# retrieve decoder weights 
decoder.layers
# Out[28]: 
# [<tensorflow.python.keras.engine.input_layer.InputLayer at 0x7f78eee3c760>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7f78eee3c0d0>,
#  <tensorflow.python.keras.layers.normalization_v2.BatchNormalization at 0x7f78eed961c0>,
#  <tensorflow.python.keras.layers.core.Dropout at 0x7f78eed96a60>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7f78eeda0940>]
# for each, the first is weights, then kernels
dec_weights = decoder.get_weights()[6] ### ==>> why is it 64x256 ???


sns.distplot(dec_weights, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Dist. decoder weights')
plt.xlabel('Weights')
plt.ylabel('Features (genes)')

out_file_name = os.path.join(outfolder, 'decoder_weights_distplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

dec_weights_dt = pd.DataFrame(dec_weights)

assert enc_weights_dt.shape[0] == dec_weights_dt.shape[1]
assert enc_weights_dt.shape[1] == dec_weights_dt.shape[0]

all_corrs = []
all_pvals = []

for i in range(enc_weights_dt.shape[0]):
        corr_, p_value = spearmanr(enc_weights_dt.iloc[i,:], 
                                   dec_weights_dt.iloc[:,i])
        all_corrs.append(corr_)
        all_pvals.append(p_value)

sns.distplot(all_corrs, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Dec. vs. enc. weights correlation')
plt.xlabel('')
plt.ylabel('Correlations')

out_file_name = os.path.join(outfolder, 'decoder_encoder_weights_correlation_distplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()


#################### 
#################### latent trasversal
#################### 
# plot the variance, those smaller than

# from cncvae.predict():
#encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
# sampling     return z_mean + K.exp(0.5 * z_log_var) * epsilon
emb_logvar_train = pred_results[1]
# 1980 x 64
emb_var_train = K.exp(emb_logvar_train).numpy()

all_mean_vars = pd.DataFrame(emb_var_train).apply(lambda x: sum(x)/len(x), axis=0)
assert len(all_mean_vars) == latent_dims
# plot histogram of mean variances
sns.distplot(all_mean_vars, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Dist. mean var of LD')
plt.xlabel('mean var')
plt.ylabel('LD')

out_file_name = os.path.join(outfolder, 'latent_dims_var_mean_distplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

sns.distplot(emb_var_train.flatten(), hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Dist. all var of LD')
plt.xlabel('var')
plt.ylabel('LD')

out_file_name = os.path.join(outfolder, 'latent_dims_var_all_distplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

# plot histogram of LD values
all_mean_zmu = pd.DataFrame(emb_train).apply(lambda x: sum(x)/len(x), axis=0)
sns.distplot(all_mean_zmu, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Dist. mean zmu of LD')
plt.xlabel('mean zmu')
plt.ylabel('LD')

out_file_name = os.path.join(outfolder, 'latent_dims_zmu_mean_distplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

sns.distplot(emb_train.flatten(), hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Dist. all zmu of LD')
plt.xlabel('var')
plt.ylabel('LD')

out_file_name = os.path.join(outfolder, 'latent_dims_zmu_all_distplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

nsamp = mrna_data_scaled.shape[0]


z_min = -4.0
z_max = 4.0
grid_nsteps = 10
z_grid = np.linspace(z_min, z_max, num=grid_nsteps)

pred_results_i = encoder.predict(pd.DataFrame(mrna_data_scaled[i,:]).T, batch_size=batch_size)[0]
pred_results_all_i = encoder.predict(mrna_data_scaled, batch_size=batch_size)[0][i,:]
sns.scatterplot(x=pred_results_i[0], y=pred_results_all_i)

pred_rec_i = decoder.predict(pd.DataFrame(emb_train[i,:]).T, batch_size=batch_size)
pred_rec_all_i = decoder.predict(emb_train, batch_size=batch_size)[i,:]
sns.scatterplot(x=pred_rec_i[0], y=pred_rec_all_i)


lds_to_traverse = [0,1,2,3]
i_samp=0
i_ld = 0
iz=z_grid[0]
# iterate over the LDs
#for i_samp in range(nsamp):

std_outputs = decoder.predict(emb_train, batch_size=batch_size)

all_traversals = dict()

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

### for a given LD, imshow of the mrna predicted by varying LD
### for a given LD, imshow of the mrna predicted by varying LD - init predicted mrna

lds_to_show = [0]
i_ld = [0]
for i_ld in lds_to_show:
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

for i_ld in lds_to_show:
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
    




ngenes = 2
gene_i = 0
nsamp=100
lds_to_traverse=[int(x) for x in all_traversals.keys()]
grid_cols = [str(x) for x in range(grid_nsteps)]
id_cols =  ['i_gene','gene','i_samp', 'sampID', 'i_LD', 'value_grid_SCC_coeff', 'value_grid_SCC_pval']
my_cols = id_cols + grid_cols

all_samp_dt = pd.DataFrame(columns = my_cols)

# if the aim is to identify a LD of interest based on gene input -> i_ld nested
# if the aim is to identify a gene of interest -> i_gene nested

i_gene=0
i_samp=0
i_ld=lds_to_traverse[0]
iz=z_grid[0]

for i_gene in range(ngenes):
    for i_samp in range(nsamp):
        for i_ld in lds_to_traverse:
            i_samp_i_gene_ldtravers = []
            for iz in z_grid:
                # retrieve the predicted matrix for the current grid value of this LD
                curr_mat = all_traversals[str(i_ld)][str(iz)] 
                assert curr_mat.shape[1] == 1000
                i_samp_i_gene_ldtravers.append(curr_mat[i_samp,i_gene])
            assert len(i_samp_i_gene_ldtravers) == grid_nsteps
            # correlation of the predicted expression with the grid value
            corr, p_val = spearmanr(i_samp_i_gene_ldtravers, z_grid)
            sns.scatterplot(x=z_grid, y =i_samp_i_gene_ldtravers)
            lt_dt = pd.DataFrame(i_samp_i_gene_ldtravers).T
            lt_dt.columns = grid_cols
            id_dt = pd.DataFrame([i_gene, gene_names[i_gene],i_samp,samp_ids[i_samp],i_ld, corr, p_val]).T
            id_dt.columns = id_cols
            ig_is_dt= pd.concat([id_dt, lt_dt], axis=1)
            all_samp_dt = pd.concat([all_samp_dt, ig_is_dt], axis=0)
        
# for a given gene, a given LD, boxplot of the predicted values along traversals
i_gene = 0
i_LD = 0
sub_dt = all_samp_dt[(all_samp_dt['i_gene'] == i_gene) & (all_samp_dt['i_LD'] == i_LD)].copy()
assert np.all(sub_dt ['i_gene'].values == i_gene)
assert np.all(sub_dt ['i_LD'].values == i_LD)
assert sub_dt.shape[0] > 0

nsamp_trav = len(set(list(sub_dt['i_samp'])))
assert sub_dt.shape[0] == nsamp_trav
# do boxplot across LD variation
box_data = sub_dt.copy()
box_data = box_data[grid_cols]
sns.boxplot(data=box_data)
x=plt.xticks(list(range(len(z_grid))), list(z_grid.round(2)))
x=plt.ylabel("Predicted gene expr.", size=12)
x=plt.xlabel("Values of LD " + str(i_LD+1), size=12)
x=plt.suptitle("Effect of LD variation on pred. expr. - " + gene_names[i_gene], size=14)
x=plt.title("(# samp = "+str(nsamp_trav)+")", size=10)
##### compare with the correlations obtained 
latent_repr = emb_train
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

# for each gene, for each LD, take the mean of the corr of expression with the traversal
gene_mean_corr_trav = all_samp_dt.groupby(['i_gene', 'i_LD']).value_grid_SCC_coeff.apply(np.mean).reset_index().copy()

corr_all_dt = pd.DataFrame(correlations_all)
corr_all_dt['i_gene'] = range(corr_all_dt.shape[0])
corr_all_m = pd.melt(corr_all_dt,id_vars=['i_gene'],var_name='i_LD', value_name='corr')

cmp_dt = gene_mean_corr_trav.merge(corr_all_m, 'inner', on=['i_gene', 'i_LD'])
sns.scatterplot(x=cmp_dt['corr'], y =cmp_dt['value_grid_SCC_coeff'])


    
#################### END
print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))

sys.exit(0)


decoder.predict(encoder.predict(mrna_data_scaled, batch_size=batch_size)[2], batch_size=batch_size)
vae.predict(mrna_data_scaled, batch_size=batch_size)

# => why it is not the same ???? => because some randomness in the data generation
# NB: this is indeed the reason why VAE useful for generating images !

#retrieve encoder weights
encoder_weights = vae.get_layer('encoder').get_layer('encoding').get_weights()[0]
encoder_weights2 = vae.get_layer('encoding').get_weights()[0]
np.all(np.equal(encoder_weights, encoder_weights2))
# for the decoder: the out layer is nested into the decoder layer, 
# so to access it should do:
decoder_weights = vae.get_layer('decoder').get_layer('out').get_weights()[0]
# decoder_weights = vae.get_layer('out').get_weights()[0] not possible !!!
assert encoder_weights.shape[0] == decoder_weights.shape[1]
assert encoder_weights.shape[1] == decoder_weights.shape[0]
#### retrieve the normal z vect
# then iteratively shut down one LD (set all values to 0)
# use it as input for the decoder
# run PCA on the decoded ^

data_with_labels = df

ndim  = 64
ncol=8
fig, axs = plt.subplots(8,8,figsize = (20,20))
palette = 'tab10'
   
for i in range(ndim):
    
    print("start LD " + str(i))
    
    i_row = math.floor((i)/ncol)
    i_col = i%ncol 
    
    intact_lds = encoder.predict(mrna_data_scaled, batch_size=batch_size)[2]
    assert intact_lds.shape[1] == ndim
    
    # set the current LD to 0, decode, and do PCA
    new_lds_dt = intact_lds
    new_lds_dt[:,i] = 0
    new_outputs = decoder.predict(new_lds_dt , batch_size=batch_size)

    mapper = umap.UMAP(n_neighbors=15, n_components=2).fit(new_outputs)
    new_umap = mapper.transform(new_outputs)
    data_to_plot = new_umap

 
    g = sns.scatterplot(data_to_plot[:,0], data_to_plot[:,1],
                        hue = list(data_with_labels['ER_Expr']), 
                        ax=axs[i_row, i_col],linewidth=0, s=15, alpha=0.7, 
                        palette = palette)
    g.set(title='LD ' + str(i+1) + ' set to 0')

    
out_file_name = os.path.join(outfolder, 'all_umap_shutdownLD_ERexpr.png')
fig.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)















### example how to subset randomly dataset
# Split 10% test set randomly
# test_set_percent = 0.1
# rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent)
# rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)

# see chapter 2 of Learn Keras for Deep Neural Networks 
# http://devpyjp.com/wp-content/uploads/2020/09/2_5300941824628622714.pdf
# Input data for a DL algorithm can be of a variety of types. Essentially, 
# the model understands data as “tensors”. Tensors are nothing but a 
# generic form for vectors, or in computer engineering terms, a simple 
# n-dimensional matrix. Data of any form is finally represented as a 
# homogeneous numeric matrix. So, if the data is tabular, it will be a two- 
# dimensional tensor where each column represents one training sample 
# and the entire table/matrix will be m samples. 
# in DL experiments, 
# it is common notation to use one training sample in a column. 
# You could also reverse the representation of training samples 
# (i.e., each row could be one training sample), so in the context of the 
# student passing/failing in the test example, one row would indicate all 
# the attributes of one student (his marks, age, etc.). 
# 
# https://keras.io/api/layers/core_layers/dense/
# Keras Dense Layer
# Just your regular densely-connected NN layer.
# 
# Dense implements the operation: output = activation(dot(input, kernel) + bias)
#  where activation is the element-wise activation function passed as the activation 
#  argument, kernel is a weights matrix created by the layer, and bias is a bias 
#  vector created by the layer (only applicable if use_bias is True). 
#  These are all attributes of Dense.
# 
# Note: If the input to the layer has a rank greater than 2, then Dense computes 
# the dot product between the inputs and the kernel along the last axis of the 
# inputs and axis 0 of the kernel (using tf.tensordot). For example, if input has
#  dimensions (batch_size, d0, d1), then we create a kernel with shape (d1, units),
#  and the kernel operates along axis 2 of the input, on every sub-tensor of shape 
#  (1, 1, d1) (there are batch_size * d0 such sub-tensors). The output in this case
#  will have shape (batch_size, d0, units).
# 
# Besides, layer attributes cannot be modified after the layer has been called 
# once (except the trainable attribute). When a popular kwarg input_shape is passed, 
# then keras will create an input layer to insert before the current layer.
#  This can be treated equivalent to explicitly defining an InputLayer.
#  
#  why tensorflow backend - from stack overflow
#  
#    1- At the beginning of Keras, the overlap with Tensorflow was small. Tensorflow
 # was a bit difficult to use, and Keras simplified it a lot.
#    2- Later, Tensorflow incorporated many functionalities similar to Keras'.
 # Keras became less necessary.
#    3- Then, apart from the multi-backend version, Keras was bundled with Tensorflow. 
# Their separation line blurred over the years.
#    4- The multi-backend Keras version was discontinued. Now the only Keras is the 
# one bundled with Tensorflow.
# 
# Update: the relationship between Keras and Tensorflow is best understood with an example:

# The dependency between Keras and Tensorflow is internal to Keras, it is not exposed 
# to the programmer working with Keras. For example, in the source code of Keras,
#  there is an implementation of a convolutional layer; this implementation calls package 
#  keras.backend to actually run the convolution computation; depending on the Keras 
#  configuration file, this backend is set to use the Tensorflow backend implementation 
#  in keras.backend.tensorflow_backend.py; this Keras file just invokes Tensorflow to compute the convolution


# https://faroit.com/keras-docs/1.2.0/backend/
# What is a "backend"?
# 
# Keras is a model-level library, providing high-level building blocks for developing deep learning models.
#  It does not handle itself low-level operations such as tensor products, convolutions and so on. 
#  Instead, it relies on a specialized, well-optimized tensor manipulation library to do so,
#  serving as the "backend engine" of Keras. Rather than picking one single tensor library and 
#  making the implementation of Keras tied to that library, Keras handles the problem in a modular way, 
#  and several different backend engines can be plugged seamlessly into Keras.
# 
# At this time, Keras has two backend implementations available: the TensorFlow backend and the Theano backend.
# 
#     TensorFlow is an open-source symbolic tensor manipulation framework developed by Google, Inc.
#     Theano is an open-source symbolic tensor manipulation framework developed by LISA/MILA Lab at Université de Montréal.



#### TYBALT
 
 # rescaled genes ranged betw 0 and 1, not the samples
 
#  2 rows × 5000 columns
# 
# rnaseq_df.apply(min,axis=0)
# 
# RPS4Y1     0.0
# XIST       0.0
# KRT5       0.0
# AGR2       0.0
# CEACAM5    0.0
#           ... 
# GDPD3      0.0
# SMAGP      0.0
# C2orf85    0.0
# POU5F1B    0.0
# CHST2      0.0
# Length: 5000, dtype: float64
# 
# rnaseq_df.apply(max,axis=0)
# 
# RPS4Y1     1.0
# XIST       1.0
# KRT5       1.0
# AGR2       1.0
# CEACAM5    1.0
#           ... 
# GDPD3      1.0
# SMAGP      1.0
# C2orf85    1.0
# POU5F1B    1.0
# CHST2      1.0
# Length: 5000, dtype: float64
# 
# rnaseq_df.apply(min,axis=1)
# 
# TCGA-02-0047-01    0.0
# TCGA-02-0055-01    0.0
# TCGA-02-2483-01    0.0
# TCGA-02-2485-01    0.0
# TCGA-02-2486-01    0.0
#                   ... 
# TCGA-ZS-A9CG-01    0.0
# TCGA-ZT-A8OM-01    0.0
# TCGA-ZU-A8S4-01    0.0
# TCGA-ZU-A8S4-11    0.0
# TCGA-ZX-AA5X-01    0.0
# Length: 10459, dtype: float64
# 
# rnaseq_df.apply(max,axis=1)
# 
# TCGA-02-0047-01    1.000000
# TCGA-02-0055-01    0.911690
# TCGA-02-2483-01    0.943803
# TCGA-02-2485-01    0.990446
# TCGA-02-2486-01    0.951420
#                      ...   
# TCGA-ZS-A9CG-01    0.994129
# TCGA-ZT-A8OM-01    1.000000
# TCGA-ZU-A8S4-01    0.987908
# TCGA-ZU-A8S4-11    0.997155
# TCGA-ZX-AA5X-01    0.912283
# Length: 10459, dtype: float64
# 
# =============================================================================
#         id_dt = pd.DataFrame({'i_gene':[i_gene], 
#                        'gene':[gene_names[i_gene]],
#                        'i_samp':[i_samp],
#                        'sampID':[ samp_ids[i_samp]]})
# =============================================================================
