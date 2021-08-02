import keras
from keras import layers
from keras.datasets import mnist
from keras import backend as K

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error,binary_crossentropy
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot

import os, sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

import datetime
start_time = str(datetime.datetime.now().time())
print('> START: mnist_vae_step_by_step.py \t' + start_time)


outfolder="BASIC_CVAE"
os.makedirs(outfolder, exist_ok=True)

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.layers import merge, concatenate
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#X_train, y_train = mnist.train.images, mnist.train.labels
#X_test, y_test = mnist.test.images, mnist.test.labels

m = 50
n_x = X_train.shape[1]
n_y = y_train.shape[1]
n_z = 2
n_epoch = 20


# Q(z|X,y) -- encoder
X = Input(batch_shape=(m, n_x))
cond = Input(batch_shape=(m, n_y))

#  how do we incorporate the new conditional variable into our existing neural net? 
# Well, let’s do the simplest thing: concatenation.


#inputs = merge([X, cond], mode='concat', concat_axis=1)
inputs = concatenate([X, cond], axis=1)

h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

# Similarly, the decoder is also concatenated with the conditional vector:

def sample_z(args):
    mu, log_sigma = args
    #eps = K.random_normal(shape=(m, n_z), mean=0., std=1.)
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X,y)
z = Lambda(sample_z)([mu, log_sigma])
#z_cond = merge([z, cond], mode='concat', concat_axis=1) # <--- NEW!
z_cond = concatenate([z, cond], axis=1) # <--- NEW!

# P(X|z,y) -- decoder
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden(z_cond)
outputs = decoder_out(h_p)

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z,y)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl


# Now the interesting part. We could generate a new data under our specific condition.
# Above, for example, we generate new data which has the label of ‘5’, i.e. c=[0,0,0,0,0,1,0,0,0,0]. 
# CVAE make it possible for us to do that.
# Things are messy here, in contrast to VAE’s Q(z|X), which nicely clusters z. But if we look at it closely, 
# we could see that given a specific value of c=y, Q(z|X,c=y) is roughly N(0,1)! It’s because, 
# if we look at our objective above, we are now modeling P(z|c), which we infer variationally with a N(0,1).
