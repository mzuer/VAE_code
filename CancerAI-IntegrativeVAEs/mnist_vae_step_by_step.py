import keras
from keras import layers
from keras.datasets import mnist
from keras import backend as K

import os, sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

import datetime
start_time = str(datetime.datetime.now().time())
print('> START: mnist_vae_step_by_step.py \t' + start_time)


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon


original_dim = 28 * 28
intermediate_dim = 64
latent_dim = 2

n_epochs = 5
batch_size = 32


outfolder = "MNIST_VAE_STEP_BY_STEP"
os.makedirs(outfolder, exist_ok=True)

outsuffix = "_" + str(n_epochs) + "epochs_" + str(batch_size) + "bs"

inputs = keras.Input(shape=(original_dim,), name="input")
h = layers.Dense(intermediate_dim, activation='relu', name="encoding")(inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(h)
z_log_sigma = layers.Dense(latent_dim, name="z_log_sigma")(h)

z = layers.Lambda(sampling, name="z")([z_mean, z_log_sigma])


# Create encoder
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu', name="decoding")(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid', name="out")(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')

outfile = os.path.join(outfolder, "encoder_modelsummary.txt")
with open(outfile, 'w') as f:
    encoder.summary(print_fn=lambda x: f.write(x + '\n'))
print("... written " + outfile)

outfile = os.path.join(outfolder, "decoder_modelsummary.txt")
with open(outfile, 'w') as f:
    decoder.summary(print_fn=lambda x: f.write(x + '\n'))
print("... written " + outfile)

outfile = os.path.join(outfolder, "modelsummary.txt")
with open(outfile, 'w') as f:
    vae.summary(print_fn=lambda x: f.write(x + '\n'))
print("... written " + outfile)

reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

#     # images / batch size = 60'000/32 = 1875
# Epoch 1/5
# 1875/1875 [==============================] - 10s 3ms/step - loss: 217.9608 - val_loss: 166.6217
# Epoch 2/5
# 1866/1875 [============================>.] - ETA: 0s - loss: 165.4826

encoder_predict = encoder.predict(x_test, batch_size=batch_size)

filename = os.path.join(outfolder, 'encoder_predict.sav')
pickle.dump(encoder_predict, open(filename, 'wb'))
print("... written " + filename)

# encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size) # I think this is wrong - not working
x_test_encoded2 = encoder.predict(x_test, batch_size=batch_size)[0] # 10000 x 2
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded2[:, 0], x_test_encoded2[:, 1], c=y_test)
plt.colorbar()
plt.show()
out_file_name = os.path.join(outfolder, "scatter_zmean.png")
plt.savefig(out_file_name, dpi=300) 
print('> saved ' + out_file_name)


# very similar if you take z instead of z_mean
x_test_encoded2 = encoder.predict(x_test, batch_size=batch_size)[2] # 10000 x 2
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded2[:, 0], x_test_encoded2[:, 1], c=y_test)
plt.colorbar()
plt.show()
out_file_name = os.path.join(outfolder, "scatter_z.png")
plt.savefig(out_file_name, dpi=300) 
print('> saved ' + out_file_name)


# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# We will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

i=0
yi=-15
j=0
xi=-15

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        #if i == 0:
         #   filename = 'mnist_stepByStep_figures/i0_decoder_predict.sav'
          #  pickle.dump(x_decoded, open(filename, 'wb'))
        # pixel values were flatten for the training
        # x_decoded is (1,784)
        # digit is (28,28), digit_size being 28
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

out_file_name = os.path.join(outfolder, "2d_manifold_latentTrasversal.png")
plt.savefig(out_file_name, dpi=300) 
print('> saved ' + out_file_name)



init_z = encoder_predict[0]
i=0
yi=grid_x[i]
for i, yi in enumerate(grid_x):
    z_sample = np.array([[yi, init_z[i][1]]]) # MZ: sampled latent vector
    x_decoded = decoder.predict(z_sample)
    #if i == 0:
     #   filename = 'mnist_stepByStep_figures/i0_decoder_predict.sav'
      #  pickle.dump(x_decoded, open(filename, 'wb'))
    # pixel values were flatten for the training
    # x_decoded is (1,784)
    # digit is (28,28), digit_size being 28
    digit = x_decoded[0].reshape(digit_size, digit_size)
#    figure[i * digit_size: (i + 1) * digit_size,
#           j * digit_size: (j + 1) * digit_size] = digit
    
    figure[i * digit_size: (i + 1) * digit_size,
           0: ( 1) * digit_size] = digit
           #j * digit_size: (j + 1) * digit_size] = digit
    

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

out_file_name = os.path.join(outfolder, "1d_manifold_latentTrasversal.png")
plt.savefig(out_file_name, dpi=300) 
print('> saved ' + out_file_name)

        
### the encoder can also be defined afterwards:
# encoder is the inference network
encoder_v2 = keras.Model(inputs, z_mean)

# a 2d plot of 10 digit classes in latent space
x_test_encoded=encoder_v2.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test)
plt.colorbar()
plt.show()

out_file_name = os.path.join(outfolder, "digits_in_latent_space.png")
plt.savefig(out_file_name, dpi=300) 
print('> saved ' + out_file_name)

#################### END
print('***** DONE\n' + start_time + " - " +  str(datetime.datetime.now().time()))

