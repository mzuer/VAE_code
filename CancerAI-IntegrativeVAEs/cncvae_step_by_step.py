#### load modules

import datetime
start_time = str(datetime.datetime.now().time())
print('> START: ' + start_time)

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

import pickle

from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding


from keras.utils.vis_utils import plot_model

#### set hard-coded params


latent_dims = 64
denselayer_size = 256 # The intermediate dense layers size
loss_distance = 'mmd'
loss_beta = 1

activ_fct = 'elu'
#args.epochs= 150 # init value ow
n_epochs= 10
batch_size = 128  
dropout_ratio = 0.2

###### load data

# training data
import pandas as pd
df=pd.read_csv(r'data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv') 

n_samp = df.shape[0]
n_genes = sum(['GE_' in x for x in df.columns])

mrna_data = df.iloc[:,34:1034].copy().values 
# the values after are CNA, the values before are clinical data
# copy() for deep copy
# values to convert to multidim array
mrna_data2= df.filter(regex='GE_').copy().values
assert mrna_data2.shape == mrna_data.shape
# 1980 x 2034
# with sample in rows, genes as columns

mrna_data.min(axis=1).shape
# (1980,)
# reshape(-1) reshapes as line vector
# reshape(-1,1) reshapes as column vector
mrna_data.min(axis=1).reshape(-1,1).shape
# (1980,1)

mrna_data.min(axis=0).shape
# 1000

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

mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ \
(mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)

# After missing-data removal, the input data sets consisted of 1000 features of
#  normalized gene expression numerical data, scaled to [0,1], and 1000 features 
# of copy number categorical data.

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
x = Dense(denselayer_size, activation=activ_fct)(inputs)
x = BN()(x)      


# ------------ Embedding Layer --------------
z_mean = Dense(latent_dims, name='z_mean')(x)
z_log_sigma = Dense(latent_dims, name='z_log_sigma', kernel_initializer='zeros')(x)
z = Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_sigma])

encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')


# Build the decoder network
# ------------ Dense out -----------------
latent_inputs = Input(shape=(latent_dims,), name='z_sampling')
x = latent_inputs
x = Dense(denselayer_size, activation=activ_fct)(x)
x = BN()(x)

x=Dropout(dropout_ratio)(x)

# ------------ Out -----------------------

#if self.args.integration == 'Clin+CNA':
#    concat_out = Dense(self.args.input_size,activation='sigmoid')(x)
#else:
concat_out = Dense(input_size)(x)

decoder = Model(latent_inputs, concat_out, name='decoder')
decoder.summary()


outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# should be done before compiling
output_model_file = os.path.join('stepByStep_figures', 'cncvae_architecture.png')
plot_model(vae, to_file=output_model_file)

output_model_file = os.path.join('stepByStep_figures', 'encoder_architecture.png')
plot_model(encoder, to_file=output_model_file)

output_model_file = os.path.join('stepByStep_figures', 'decoder_architecture.png')
plot_model(decoder, to_file=output_model_file)

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
vae.compile(optimizer=adam)
vae.summary()



with open('stepByStep_figures/modelsummary.txt', 'w') as f:
    vae.summary(print_fn=lambda x: f.write(x + '\n'))

s_train = mrna_data_scaled
s_test = mrna_data_scaled

ds_train = s_train#np.concatenate((s1_train,s2_train), axis=-1)
ds_test = s_test#np.concatenate((s1_test,s2_test), axis=-1)

# from cncvae.train():
vae.fit(ds_train, epochs=n_epochs, batch_size=batch_size, shuffle=True, validation_data=(ds_test, None))
        
# from cncvae.predict():
emb_train = encoder.predict(mrna_data_scaled, batch_size=batch_size)[0]


print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))