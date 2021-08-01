
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

import tensorflow as tf

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
concat_out = Dense(input_size, name="out")(x)

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
history = vae.fit(ds_train, epochs=n_epochs, batch_size=batch_size, shuffle=True, validation_data=(ds_test, None))
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
plt.show()

filename = os.path.join(outfolder, 'train_valid_loss'+ outsuffix +'.png')
plt.savefig(filename, dpi=300) 
print('... written: ' + filename)


#################### look at the weights

#vae_weights = model.load(vae_weights_150epochs_128bs.h5
# from keras.models import load_model
# vae_loaded = load_model('CNCVAE_STEP_BY_STEP/vae_150epochs_128bs.h5')

vae_loaded = vae

all_weights = dict()

i=0
for layer in vae_loaded.layers:
  print(i)
  i+=1
  print(layer.name)
  print("Weights")
  if len(layer.get_weights()) > 0: 
      print("Shape: ",layer.get_weights()[0].shape,'\n')
      all_weights[layer.name] = layer.get_weights()[0]

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
plt.title('Histogram of corr values')
plt.xlabel('Weights')
plt.ylabel('Features (genes)')

out_file_name = os.path.join(outfolder, 'encoder_weights_distplot.png')
plt.savefig(out_file_name, dpi=300) 
print('... written: ' + out_file_name)

# find the gene that has the max activation
enc_weights_dt = pd.DataFrame(all_weights['encoding'])
maxvals = enc_weights_dt.apply(max, axis=1)
np.argmax(maxvals)

df.columns[34:1034][np.argmax(maxvals)]
# The transcription factor DEC1 (stra13, SHARP2) is associated with the 
# hypoxic response and high tumour grade in human breast cancers

      
#################### END
print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))


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

