
wd = os.path.join('/home','marie','Documents','FREITAS_LAB','VAE_tutos','CancerAI-IntegrativeVAEs')
os.chdir(wd)

modelRunFolder = os.path.join('CNCVAE_STEP_BY_STEP')

outfolder = 'VIZ_CNCVAE_STEP_BY_STEP'
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

# raw data 
#df=pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\data\MBdata_33CLINwMiss_1KfGE_1KfCNA.csv') # dataset available in github repo
df=pd.read_csv(os.path.join('data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv')) # d

#np.savetxt(r"C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\code\IntegrativeVAEs\code\results\custom_arch\mRNA_ls64_hs256_mmd_beta1_scaled.csv", emb_train, delimiter = ',')
outfile = os.path.join(outfolder, "mRNA_ls64_hs256_mmd_beta1_scaled" + outsuffix + ".csv")
np.savetxt(outfile, emb_train, delimiter = ',')



