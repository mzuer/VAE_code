
# Rscript vae_LS_downstream.R

require(keras)

mainfolder <- file.path('/home','marie','Documents','FREITAS_LAB','VAE_tutos','CancerAI-IntegrativeVAEs')

modelRunFolder <- file.path(mainfolder, 'CNCVAE_STEP_BY_STEP')

outfolder <- 'VAE_LS_DOWNSTREAM'
dir.create(outfolder, recursive = T)

# for sanity check
nfeatures <- 1000
nsamp <- 1980

# for loading data
n_epochs <- 150
batch_size <- 128  
outsuffix <-  paste0("_" , n_epochs, "epochs_" ,batch_size,  "bs")


# load raw data for the labels
# load the latent representation
# try some metric for differentiating if good clustering




### load the model
vae_model_file <- file.path(modelRunFolder, paste0("vae_", n_epochs, "epochs_", batch_size, "bs.h5"))
vae <- load_model_hdf5(vae_model_file)

all_layer_names <- unlist(lapply(vae$layers, function(x) unlist(x)$name))
for(i in all_layer_names){
  cat("> For layer ", i, "\n")
  layer_wb <- get_weights(get_layer(vae, i))
  print(lapply(layer_wb, dim))
}

### the decoder layers are nested within decoder !!!
get_layer(vae, "decoder")$layers
all_decoder_layer_names <- unlist(lapply( get_layer(vae, "decoder")$layers, function(x) unlist(x)$name))
  
# raw input data
raw_file <- file.path(mainfolder, 'data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv')
raw_data <- read.delim(raw_file, sep="\t")

nsamp <- nrow(raw_data)
stopifnot(nsamp == nsamp)

mrna_data <- raw_data[,grep("^GE_", colnames(raw_data))]
stopifnot(ncol(mrna_data) == nfeatures)

mrna_data <-  df.iloc[:,34:1034].copy().values 

# latent dimension data
latent_dim_file <- file.path(modelRunFolder, paste0("mRNA_ls64_hs256_mmd_beta1_scaled", outsuffix, ".csv"))
ld_dt <- read.delim(latent_dim_file, header=F, sep = ',')
# should be same as
emb_train_file <- file.path(os.path.join(modelRunFolder,'emb_train'+outsuffix+'.sav'), 'rb')
emb_train  = pickle.load(file)


#### take gene set from kegg pathway
# do they have high correlation with one of the LD
# if so, maybe interested to look at the other genes that have high correlation within this LD


#### gene enrichment analysis
