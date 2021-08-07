


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras

import seaborn as sns

#import pydot
#import graphviz
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model

from keras_tqdm import TQDMNotebookCallback
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# -

import datetime
start_time =  str(datetime.datetime.now().time())

outfolder = "TYBALT_VAE"
os.makedirs(outfolder, exist_ok=True)
#tf.random.set_seed(42)



inputfolder=os.path.join("../Tybalt/tybalt-master", "data")

print(keras.__version__)
tf.__version__

#plt.style.use('seaborn-notebook')

#sns.set(style="white", color_codes=True)
#sns.set_context("paper", rc={"font.size":14,"axes.titlesize":15,"axes.labelsize":20,
#                             'xtick.labelsize':14, 'ytick.labelsize':14})

#import tensorflow.compat.v1.keras.backend as K
#tf.compat.v1.disable_eager_execution()

# Load Functions and Classes
#
# This will facilitate connections between layers and also custom hyperparameters


# Function for reparameterization trick to make model differentiable
def sampling(args):
    
    import tensorflow as tf
    # Function with args required for Keras Lambda function
    z_mean, z_log_var = args

    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)
    
    # The latent vector is non-deterministic and differentiable
    # in respect to z_mean and z_log_var
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z


class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training
    This function is borrowed from:
    https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
    """
    def __init__(self, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - 
                                K.exp(z_log_var_encoded), axis=-1)
        return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


# ### Implementing Warm-up as described in Sonderby et al. LVAE
#
# This is modified code from https://github.com/fchollet/keras/issues/2595

class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


np.random.seed(123)

# ## Load Gene Expression Data

rnaseq_file = os.path.join(inputfolder, 'pancan_scaled_zeroone_rnaseq.tsv.gz')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
print(rnaseq_df.shape)
rnaseq_df.head(2)

# Split 10% test set randomly
test_set_percent = 0.1
rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent)
rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)

# ## Initialize variables and hyperparameters

# Set hyper parameters
original_dim = rnaseq_df.shape[1]
latent_dim = 100

batch_size = 50
epochs = 3 # init value: 50
learning_rate = 0.0005

epsilon_std = 1.0
beta = K.variable(0)
kappa = 1

outsuffix = str(latent_dim) + "LD_" + str(epochs) + "epochs_" + str(batch_size) + "bs"

# ## Encoder

# Input place holder for RNAseq data with specific input size
rnaseq_input = Input(shape=(original_dim, ))

# Input layer is compressed into a mean and log variance vector of size `latent_dim`
# Each layer is initialized with glorot uniform weights and each step (dense connections,
# batch norm, and relu activation) are funneled separately
# Each vector of length `latent_dim` are connected to the rnaseq input tensor
z_mean_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

z_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

# return the encoded and randomly sampled z vector
# Takes two keras layers as input to the custom sampling function layer with a `latent_dim` output
z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean_encoded, z_log_var_encoded])
# -

# ## Decoder

# The decoding layer is much simpler with a single layer and sigmoid activation
decoder_to_reconstruct = Dense(original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
rnaseq_reconstruct = decoder_to_reconstruct(z)

# ## Connect the encoder and decoder to make the VAE
#
# The `CustomVariationalLayer()` includes the VAE loss function (reconstruction + (beta * KL)), which is what will drive our model to learn an interpretable representation of gene expression space.
#
# The VAE is compiled with an Adam optimizer and built-in custom loss function. The `loss_weights` parameter ensures beta is updated at each epoch end callback

adam = optimizers.Adam(lr=learning_rate)
vae_layer = CustomVariationalLayer()([rnaseq_input, rnaseq_reconstruct])
vae = Model(rnaseq_input, vae_layer)
vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

vae.summary()


# Visualize the connections of the custom VAE model
output_model_file = os.path.join(outfolder, 'onehidden_vae_architecture.png')
plot_model(vae, to_file=output_model_file)
plt.close()
print('... written: ' + output_model_file)
#SVG(model_to_dot(vae).create(prog='dot', format='svg'))


# ## Train the model
#
# The training data is shuffled after every epoch and 10% of the data is heldout for calculating validation loss.

#tf.compat.v1.global_variables_initializer()

hist = vae.fit(np.array(rnaseq_train_df),
               shuffle=True,
               epochs=epochs,
               verbose=0,
               batch_size=batch_size,
               validation_data=(np.array(rnaseq_test_df), None),
                              callbacks=[WarmUpCallback(beta, kappa)])
#               callbacks=[WarmUpCallback(beta, kappa),
#                          TQDMNotebookCallback(leave_inner=True, leave_outer=True)])

# Visualize training performance
history_df = pd.DataFrame(hist.history)
hist_plot_file = os.path.join(outfolder, 'onehidden_vae_training.pdf')
ax = history_df.plot()
ax.set_xlabel('Epochs')
ax.set_ylabel('VAE Loss')
fig = ax.get_figure()
fig.savefig(hist_plot_file)
plt.close()
print('... written: ' + hist_plot_file)

# ## Compile and output trained models
#
# We are interested in:
#
# 1. The model to encode/compress the input gene expression data
#   * Can be possibly used to compress other tumors
# 2. The model to decode/decompress the latent space back into gene expression space
#   * This is our generative model
# 3. The latent space compression of all pan cancer TCGA samples
#   * Non-linear reduced dimension representation of tumors can be used as features for various tasks
#     * Supervised learning tasks predicting specific gene inactivation events
#     * Interpolating across this space to observe how gene expression changes between two cancer states
# 4. The weights used to compress each latent node
#   * Potentially indicate learned biology differentially activating tumors

# ### Encoder model

# Model to compress input
encoder = Model(rnaseq_input, z_mean_encoded)

# +
# Encode rnaseq into the hidden/latent representation - and save output
encoded_rnaseq_df = encoder.predict_on_batch(rnaseq_df)
encoded_rnaseq_df = pd.DataFrame(encoded_rnaseq_df, index=rnaseq_df.index)

encoded_rnaseq_df.columns.name = 'sample_id'
encoded_rnaseq_df.columns = encoded_rnaseq_df.columns + 1
encoded_file = os.path.join(outfolder, 'encoded_rnaseq_onehidden_warmup_batchnorm.tsv')
encoded_rnaseq_df.to_csv(encoded_file, sep='\t')
# -

# ### Decoder (generative) model

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim, ))  # can generate from any sampled z vector
_x_decoded_mean = decoder_to_reconstruct(decoder_input)
decoder = Model(decoder_input, _x_decoded_mean)

# ## Save the encoder/decoder models for future investigation

# +
encoder_model_file = os.path.join(outfolder, 'encoder_onehidden_vae.hdf5')
decoder_model_file = os.path.join(outfolder, 'decoder_onehidden_vae.hdf5')

encoder.save(encoder_model_file)
decoder.save(decoder_model_file)
# -

# ##  Model Interpretation - Sanity Check
#
#
# ###  Observe the distribution of node activations.
#
# We want to ensure that the model is learning a distribution of feature activations, and not zeroing out features.

# +
# What are the most and least activated nodes
# encoded_rnaseq_df = predictions of the encoder
sum_node_activity = encoded_rnaseq_df.sum(axis=0).sort_values(ascending=False)

# Top 10 most active nodes
print(sum_node_activity.head(10))

# Bottom 10 least active nodes
sum_node_activity.tail(10)
# -

# Histogram of node activity for all 100 latent features
sum_node_activity.hist()
plt.xlabel('Activation Sum')
plt.ylabel('Count');
out_file_name = os.path.join(outfolder, 'node_activity_LF_hist' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

# What does an example distribution of two latent features look like?

# Example of node activation distribution for the first two latent features
plt.figure(figsize=(6, 6))
plt.scatter(encoded_rnaseq_df.iloc[:, 1], encoded_rnaseq_df.iloc[:, 2])
plt.xlabel('Latent Feature 1')
plt.xlabel('Latent Feature 2');
out_file_name = os.path.join(outfolder, 'node_activation_LD1_LD2' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()

# ###  Observe reconstruction fidelity

# How well does the model reconstruct the input RNAseq data
input_rnaseq_reconstruct = decoder.predict(np.array(encoded_rnaseq_df))
input_rnaseq_reconstruct = pd.DataFrame(input_rnaseq_reconstruct, index=rnaseq_df.index,
                                        columns=rnaseq_df.columns)
input_rnaseq_reconstruct.head(2)

# +
reconstruction_fidelity = rnaseq_df - input_rnaseq_reconstruct

gene_mean = reconstruction_fidelity.mean(axis=0)
gene_abssum = reconstruction_fidelity.abs().sum(axis=0).divide(rnaseq_df.shape[0])
gene_summary = pd.DataFrame([gene_mean, gene_abssum], index=['gene mean', 'gene abs(sum)']).T
gene_summary.sort_values(by='gene abs(sum)', ascending=False).head()
# -

# Mean of gene reconstruction vs. absolute reconstructed difference per sample
g = sns.jointplot('gene mean', 'gene abs(sum)', data=gene_summary, stat_func=None);
out_file_name = os.path.join(outfolder, 'reconstruct_fidelity_mean_abssum' + outsuffix + '.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)
plt.close()


#################### END
print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))




