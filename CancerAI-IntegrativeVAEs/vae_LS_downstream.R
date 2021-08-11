
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

encoder_model_file <- file.path(modelRunFolder, paste0("encoder_", n_epochs, "epochs_", batch_size, "bs.h5"))
encoder <- load_model_hdf5(encoder_model_file)


# raw input data
raw_file <- file.path(mainfolder, 'data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv')
raw_data <- read.delim(raw_file, sep=",")

nsamp <- nrow(raw_data)
stopifnot(nsamp == nsamp)

mrna_data <- raw_data[,grep("^GE_.+", colnames(raw_data))]
stopifnot(ncol(mrna_data) == nfeatures)


# rescale to have samples ranging from 0 to 1 [by row0]
mrna_data_scaled <- t(apply(mrna_data, 1, function(x) (x-min(x))/(max(x) - min(x))))
stopifnot(dim(mrna_data_scaled) == dim(mrna_data))
foo <- apply(mrna_data_scaled, 1, function(x) stopifnot(range(x) == c(0,1)))

emb_train <- data.frame(predict_on_batch(encoder, mrna_data_scaled)[[1]])
emb_logsigma <- data.frame(predict_on_batch(encoder, mrna_data_scaled)[[2]])

# latent dimension data
latent_dim_file <- file.path(modelRunFolder, paste0("mRNA_ls64_hs256_mmd_beta1_scaled_embtrain", outsuffix, ".csv"))
ld_dt <- read.delim(latent_dim_file, header=F, sep = ',')

#not exactly the same -> some diff. R / python

stopifnot(abs(emb_train-ld_dt) < 10^-5)

############# find for which clinical data the highest diff in CLES
# clin variable at the top right
get_CLES <- function(catvar, ref_cat, LS_with_clinic_dt, other_cat=NULL, filtvars = c("?")) {
  my_var <- LS_with_clinic_dt[,catvar]
  cat_samples <- which(my_var %in% ref_cat)
  if(is.null(other_cat)){
    other_samples <- which(!my_var %in% ref_cat & !my_var %in% filtvars & !is.na(my_var))
  }else {
    other_samples <- which(my_var %in% other_cat & !my_var %in% filtvars & !is.na(my_var))
  }
  stopifnot(length(cat_samples) > 0)
  stopifnot(length(other_samples) > 0)
  start_clin_idx <- which(colnames(LS_with_clinic_dt) =="METABRIC_ID")
  LD_CLES <- apply(LS_with_clinic_dt[,1:(start_clin_idx-1)], 2, function(x) {
    CLES(x[cat_samples], x[other_samples], distribution = NA)
  })
  stopifnot(length(LD_CLES) == start_clin_idx-1)
  LD_CLES
}


mycatvar <- "Pam50Subtype"
myref <- "Her2"
train_annot_dt <- cbind(ld_dt, raw_data[,c("METABRIC_ID", mycatvar)]) 
LD_CLES <- get_CLES(mycatvar, myref, train_annot_dt)
stopifnot(length(LD_CLES) == ncol(ld_dt))

# find the most associated
LD_CLES_05 <- LD_CLES-0.5

# the highest pos
LD_topCLES_1pos <- which.max(LD_CLES_05)

# the highest neg
LD_topCLES_1neg <- which.min(LD_CLES_05)


get_ld_gene_corr <- function(mrnadata, ld_vect) {
  apply(mrna_data_scaled, 2, function(x){
    cor(x=ld_vect, y=x)
  })
}

topld_genecorrs <- get_ld_gene_corr(mrna_data_scaled, ld_dt[, LD_topCLES_1pos])

# have an idea of the corrs to set the thresh
plot(dist(topld_genecorrs))

quantile(abs(topld_genecorrs))

get_top_genes <- function(corr_vect, corr_thresh, removePattern=NULL) {
  top_pos_genes <- names(corr_vect)[corr_vect >= abs(corr_thresh)]
  top_neg_genes <- names(corr_vect)[corr_vect <= -abs(corr_thresh)]
  if(!is.null(removePattern)) {
    top_pos_genes <- gsub(removePattern, "", top_pos_genes )  
    top_neg_genes <- gsub(removePattern, "", top_neg_genes)  
  }
  return(list(
    top_pos_genes=top_pos_genes,
    top_neg_genes=top_neg_genes
  ))
}

highcorr_thresh <- as.numeric(quantile(abs(topld_genecorrs), probs=0.75))
topld_topcorrgenes <- get_top_genes(topld_genecorrs, 0.2, removePattern="GE_")




enrichResult1 <- WebGestaltR(enrichMethod="ORA", organism="hsapiens",
                            minNum=4,
                            enrichDatabase="pathway_KEGG", interestGene=topld_topcorrgenes[["top_pos_genes"]],
                            interestGeneType="genesymbol", referenceSet="genome",
                            referenceGeneType="genesymbol", isOutput=F,
                            sigMethod = "top", # default: fdr
                            fdrMethod = "BH")# default
enrichResult1$description

enrichResult2 <- WebGestaltR(enrichMethod="ORA", organism="hsapiens",
                            minNum=4,
                            enrichDatabase="pathway_KEGG", interestGene=topld_topcorrgenes[["top_pos_genes"]],
                            interestGeneType="genesymbol", referenceSet="genome_protein-coding",
                            referenceGeneType="genesymbol", isOutput=F,
                            sigMethod = "top", # default: fdr
                            fdrMethod = "BH")# default
enrichResult2$description


enrichResult3 <- WebGestaltR(enrichMethod="ORA", organism="hsapiens",
                            minNum=4,
                            enrichDatabase="geneontology_Biological_Process_noRedundant", interestGene=topld_topcorrgenes[["top_pos_genes"]],
                            interestGeneType="genesymbol", referenceSet="genome_protein-coding",
                            referenceGeneType="genesymbol", isOutput=F,
                            sigMethod = "top", # default: fdr
                            fdrMethod = "BH")# default
enrichResult3$description


"network_TCGA_RNASeq_BRCA"

enrichResult4 <- WebGestaltR(enrichMethod="ORA", organism="hsapiens",
                             minNum=4,
                             enrichDatabase="network_TCGA_RNASeq_BRCA", interestGene=topld_topcorrgenes[["top_pos_genes"]],
                             interestGeneType="genesymbol", referenceSet="genome_protein-coding",
                             referenceGeneType="genesymbol", isOutput=F,
                             sigMethod = "top", # default: fdr
                             fdrMethod = "BH")# default
enrichResult4$description



enrichResult5 <- WebGestaltR(enrichMethod="ORA", organism="hsapiens",
                             minNum=4,
                             enrichDatabase="pathway_Wikipathway_cancer", interestGene=topld_topcorrgenes[["top_pos_genes"]],
                             interestGeneType="genesymbol", referenceSet="genome_protein-coding" ,
                             referenceGeneType="entrezgene", isOutput=F,
                             sigMethod = "top", # default: fdr
                             fdrMethod = "BH")# default
enrichResult5$description


all_gene_corrs_abs<- abs(all_gene_corrs)
sum(all_gene_corrs_abs >= highcorr_thresh)
# 56
corr_genes <- names(all_gene_corrs_abs)[all_gene_corrs_abs >= highcorr_thresh]
stopifnot(grepl("^GE_", corr_genes))
corr_genes <- gsub("^GE_", "", corr_genes)








get_CLES("ER_Status", "pos", train_annot_dt, other_cat="neg")

catvar="ER_Status"
train_annot_dt <- cbind(ld_dt, raw_data[,c("METABRIC_ID", catvar)]) 
train_annot_dt <- train_annot_dt[train_annot_dt[,catvar] %in% c("pos", "neg"),]
pos_samples <- which(train_annot_dt$ER_Status == "pos")
neg_samples <- which(train_annot_dt$ER_Status == "neg")

# 1938
LD_CLES <- apply(train_annot_dt[,1:ncol(ld_dt)], 2, function(x) {
  CLES(x[pos_samples], x[neg_samples], distribution = NA)
})
stopifnot(length(LD_CLES) == ncol(ld_dt))
LD_topCLES_1b <- which.max(LD_CLES)

### retrieve the genes that highly correlate with it
all_gene_corrs <- apply(mrna_data_scaled, 2, function(x){
  cor(x=ld_dt[, LD_topCLES_1b], y=x)
})
all_gene_corrs_abs <- abs(all_gene_corrs)
highcorr_thresh <- 0.4
sum(all_gene_corrs_abs >= highcorr_thresh)
# 56
corr_genes <- names(all_gene_corrs_abs)[all_gene_corrs_abs >= highcorr_thresh]
stopifnot(grepl("^GE_", corr_genes))
corr_genes <- gsub("^GE_", "", corr_genes)


################# only 1000 other genes, use full genome as reference
outFolder <- file.path("VAES_LS_DOWNSTREAM")
dir.create(outFolder, recursive=T)
library(httr)
organism <- "hsapiens"
referenceSet <- "genome_protein-coding"
response <- GET("http://www.webgestalt.org/api/reference",
                query=list(organism=organism, referenceSet=referenceSet))
if (response$status_code == 200) {
  fileContent <- content(response)
  ref_outfile <- file.path(outFolder, paste0(organism, "_", referenceSet, "_reference.txt"))
  write(fileContent, outfile)
  genes <- unlist(strsplit(fileContent, "\n", fixed=TRUE))
  print(genes[1:3])
}











#### take gene set from kegg pathway
# do they have high correlation with one of the LD
# if so, maybe interested to look at the other genes that have high correlation within this LD


#### gene enrichment analysis
catvar <- "ER_Status"
train_annot_dt <- cbind(ld_dt, raw_data[,c("METABRIC_ID", catvar)])
train_annot_dt <- train_annot_dt[train_annot_dt[,catvar] %in% c("pos", "neg"),]
pos_samples <- which(train_annot_dt$ER_Status == "pos")
neg_samples <- which(train_annot_dt$ER_Status == "neg")

# 1938
LD_CLES <- apply(train_annot_dt[,1:ncol(ld_dt)], 2, function(x) {
  CLES(x[pos_samples], x[neg_samples], distribution = NA)
})
stopifnot(length(LD_CLES) == ncol(ld_dt))
LD_topCLES_1b <- which.max(LD_CLES)

### retrieve the genes that highly correlate with it
all_gene_corrs <- apply(mrna_data_scaled, 2, function(x){
  cor(x=ld_dt[, LD_topCLES_1b], y=x)
})
all_gene_corrs_abs <- abs(all_gene_corrs)
highcorr_thresh <- 0.4
sum(all_gene_corrs_abs >= highcorr_thresh)
# 56
corr_genes <- names(all_gene_corrs_abs)[all_gene_corrs_abs >= highcorr_thresh]
stopifnot(grepl("^GE_", corr_genes))
corr_genes <- gsub("^GE_", "", corr_genes)


################# only 1000 other genes, use full genome as reference
outFolder <- file.path("VAES_LS_DOWNSTREAM")
dir.create(outFolder, recursive=T)
library(httr)
organism <- "hsapiens"
referenceSet <- "genome_protein-coding"
response <- GET("http://www.webgestalt.org/api/reference",
                query=list(organism=organism, referenceSet=referenceSet))
if (response$status_code == 200) {
  fileContent <- content(response)
  ref_outfile <- file.path(outFolder, paste0(organism, "_", referenceSet, "_reference.txt"))
  write(fileContent, outfile)
  genes <- unlist(strsplit(fileContent, "\n", fixed=TRUE))
  print(genes[1:3])
}



enrichWGR_df <- WebGestaltR::WebGestaltR(
  enrichMethod = "ORA",
  enrichDatabase="pathway_KEGG",
  organism = "hsapiens",
  interestGene = corr_genes,
  interestGeneType = "genesymbol",
  minNum = 4,
  sigMethod ="top",  #fdr", # default
  fdrMethod = "BH",# default
  isOutput = F,
 # outputDirectory = out_dir,
 referenceGeneFile = ref_outfile,
  referenceGeneType = "entrezgene_protein-coding"
)



enrichWGR_df2 <- WebGestaltR::WebGestaltR(
  enrichMethod = "ORA",
  enrichDatabase="pathway_KEGG",
  organism = "hsapiens",
  interestGene = corr_genes,
  interestGeneType = "genesymbol",
  minNum = 4,
  sigMethod ="top",  #fdr", # default
  fdrMethod = "BH",# default
  isOutput = F,
  # outputDirectory = out_dir,
  referenceGene = gsub("^GE_", "", names(all_gene_corrs_abs)),
  referenceGeneType = "genesymbol"
)

ref_genes <- names(all_gene_corrs_abs)[all_gene_corrs_abs < highcorr_thresh]
stopifnot(grepl("^GE_", ref_genes))
ref_genes <- gsub("^GE_", "", ref_genes)

enrichResult <- WebGestaltR(enrichMethod="ORA", organism="hsapiens",
                            minNum=4,
                            enrichDatabase="pathway_KEGG", interestGene=corr_genes,
                            interestGeneType="genesymbol", referenceSet="genome",
                            referenceGeneType="genesymbol", isOutput=F,
                            sigMethod = "top", # default
                            fdrMethod = "BH",# default
)




webgestalt_df <- WebGestaltR::WebGestaltR(
  enrichMethod = "ORA",
  enrichDatabase="pathway_KEGG",
  organism = "hsapiens",
  interestGeneFile = geneFile,
  interestGeneType = "genesymbol",
  minNum = 4,
  sigMethod = "fdr", # default
  fdrMethod = "BH",# default
  isOutput = TRUE,
  # outputDirectory = out_dir,
  referenceGeneFile = outfile,
  referenceGeneType = "entrezgene_protein-coding"
)


############## cmp the umap

# run umap on the raw data
# in python mapper = umap.UMAP(n_neighbors=15, n_components=2).fit(mrna_data)
library(umap)

raw_umap <- umap(mrna_data, n_neighbors=15, n_components=2)
plot(raw_umap$layout[,1],raw_umap$layout[,2])

scaled_umap <- umap(mrna_data_scaled, n_neighbors=15, n_components=2)
plot(scaled_umap$layout[,1],scaled_umap$layout[,2])

# first umap on raw data

py_umap_raw <- read.delim("VIZ_CNCVAE_STEP_BY_STEP/umap_coord_rawLDs_150epochs_128bs.csv", sep=",", header=F)
plot(py_umap_raw[,1], py_umap_raw[,2])


################################# ################################# 
cat("***** DONE\n")
cat(paste0(Sys.time(), "\n"))


################################# 
################################# 
################################# TRASH


all_layer_names <- unlist(lapply(vae$layers, function(x) unlist(x)$name))
for(i in all_layer_names){
  cat("> For layer ", i, "\n")
  layer_wb <- get_weights(get_layer(vae, i))
  print(lapply(layer_wb, dim))
}

### the decoder layers are nested within decoder !!!
get_layer(vae, "decoder")$layers
all_decoder_layer_names <- unlist(lapply( get_layer(vae, "decoder")$layers, function(x) unlist(x)$name))


####################### THRASH DE analysis


# for the design:
my_group <- unlist(sapply(colnames(exprDT), function(x) {
  ifelse(x %in% samp1, cond1, ifelse(x %in% samp2, cond2, NA))
}))
stopifnot(!any(is.na(my_group)))
stopifnot(length(my_group) == ncol(exprDT))

# design matrix, cond1 as reference (for voom())
my_group_design <- factor(my_group, levels = c(cond1, cond2))
my_design <- model.matrix( ~ my_group_design)


seqData <- DGEList(exprDT, group=my_group, genes = rownames(exprDT))
seqData <- calcNormFactors(seqData)
outFile <- paste0(curr_outFold, "/", "mean_variance_voom.png")
png(paste0(outFile))
voomData <- voom(seqData, design = my_design, plot=T)
foo <- dev.off()
cat(paste0("... written: ", outFile, "\n"))
vfitData <- lmFit(voomData, voomData$design)
efitData <- eBayes(vfitData)


