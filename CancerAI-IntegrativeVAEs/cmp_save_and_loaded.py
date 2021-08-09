# python cmp_save_and_loaded.py

import datetime
start_time = str(datetime.datetime.now().time())
print('> START: cmp_save_and_loaded.py \t' + start_time)

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

outfolder = 'CMP_SAVE_AND LOADED'
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
mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ \
(mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)


# compare emb_train to what we obtain if reload the model
from keras.models import load_model
vae_loaded = load_model('CNCVAE_STEP_BY_STEP/vae_150epochs_128bs.h5')
vae=vae_loaded

#encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
encoder_loaded = load_model('CNCVAE_STEP_BY_STEP/encoder_150epochs_128bs.h5')

pred_results = encoder_loaded.predict(mrna_data_scaled, batch_size=batch_size)
emb_train2 = pred_results[0]
from scipy.stats import spearmanr

all_corrs = []
for i in range(emb_train.shape[1]):
        corr_, p_value = spearmanr(emb_train[:,i], emb_train2[:,i])
        all_corrs.append(corr_)

assert np.min(all_corrs) == np.max(all_corrs)
assert np.min(all_corrs) == 1

sns.scatterplot(emb_train[:,0], emb_train2[:,0])
