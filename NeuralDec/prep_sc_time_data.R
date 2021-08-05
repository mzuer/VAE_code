
# Rscript pre_sc_time_data.R


startTime <- Sys.time()

outFolder <- file.path("PREP_SC_TIME_DATA")

# Märtens et al. 2019
# We applied ND to 820 cells using the 7,500 most vari-able genes, following the previously published linear
# analysis

# from: https://kieranrcampbell.github.io/phenopath/phenopath_shalek_vignette.html#getting-the-data-ready-for-phenopath

# if needed
# install.packages("devtools")
# install.packages("BiocManager")
#require(devtools)
#require(BiocManager)
# BiocManager::install("scater")
#BiocManager::install("MultiAssayExperiment")
library(MultiAssayExperiment)
# download data from wget http://imlspenticton.uzh.ch/robinson_lab/conquer/data-mae/GSE48968-GPL13112.rds
mae <- readRDS("phenopath_data/GSE48968-GPL13112.rds")

# Next we’re going to retrieve counts, transcript-per-million (TPM) values and the phenotypic (cell-specific) data 
# and convert it into an SCESet used by scater. We’ll set the default “expression” values to log2(TPM+1)

# install.packages("scater")
suppressPackageStartupMessages(library(scater))
cts <- assays(experiments(mae)[["gene"]])[["count_lstpm"]]
tpms <- assays(experiments(mae)[["gene"]])[["TPM"]]
phn <- colData(mae)

# sce <- newSCESet(countData = cts, 
#                  phenoData = new("AnnotatedDataFrame", data = as.data.frame(phn)))
# newSCEset was removed from latest version of scater !
cat("> Build SingleCellExperiment\n")
sce <- SingleCellExperiment(countData = cts, 
                            phenoData = new("AnnotatedDataFrame", data = as.data.frame(phn)))
tpm(sce) <- tpms
exprs(sce) <- log2(tpm(sce) + 1)

# We’re only interested in cells exposed to LPS or PAM, so we parse these from
# the description column of the SCESet and subset the data accordingly:

is_lps_pam <- grepl("LPS|PAM", sce$description)
sce <- sce[, is_lps_pam]

# Finally, we need to parse the capture time and stimulant from the description column of the SCESet and add them as new columns:

split <- strsplit(as.character(sce$description), "_", fixed = TRUE)
stimulant <- sapply(split, `[`, 1)
time <- sapply(split, `[`, 2)
sce$stimulant <- stimulant
sce$time <- time

# Finally, let’s get MGI symbols for the genes so we actually know what they are:

suppressPackageStartupMessages(library(biomaRt))
ensembl_gene_ids <- sapply(strsplit(featureNames(sce), ".", fixed = TRUE), `[`, 1)
mart <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")
bm <- getBM(attributes = c("ensembl_gene_id", "mgi_symbol"),
            filters = "ensembl_gene_id",
            values = ensembl_gene_ids,
            mart = mart)

fData(sce)$mgi_symbol <- rep(NA, nrow(sce))

mm2 <- match(bm$ensembl_gene_id, ensembl_gene_ids)
fData(sce)$mgi_symbol[mm2] <- bm$mgi_symbol


########### QUALITY CONTROL

# The next stage is quality control and removal of low-quality cells.
# We begin by calling the scater function calculateQCMetrics:

cat("> calculateQCMetrics\n")
sce <- calculateQCMetrics(sce)

# We can plot the total number of genes expressed (total_features) against the total number of counts to each cell:

plotPhenoData(sce, aes(x = total_counts, y = total_features))


# We see there are quite a few cells with low counts and features.
# We’ll remove these via threholds:

sce$to_keep <- sce$total_counts > 5e5 & sce$total_features > 5e3
plotPhenoData(sce, aes(x = total_counts, y = total_features, color = to_keep)) +
  labs(subtitle = "QC pass: total_features > 5000 and total_counts > 50000")

# and subset to the post-qc’d cells:

sce_qc <- sce[, sce$to_keep]

# In the original publication (Shalek et al. (2014)) the author identified a subset of “cluster-disrupted”
# cells that were removed. These were identified as having low Lyz1 expression and high Serpinb6b expression. 
# Let’s have a look at the co-expression of these two:

Lyz1_index <- grep("Lyz1", fData(sce_qc)$mgi_symbol)
SerpinB6b_index <- grep("SerpinB6b", fData(sce_qc)$mgi_symbol, ignore.case = TRUE)

Lyz1 <- exprs(sce_qc)[Lyz1_index,]
Serpinb6b <- exprs(sce_qc)[SerpinB6b_index,]

qplot(Lyz1, Serpinb6b)

# Accepting cells with Lyz1 expression greater than 0 and Serpbinb6b expression less than 2.5 seems reasonable. 
# Let’s see how this would look:

Serpinb6b_threshold <- 2.5
Lyz1_threshold <- 0

to_keep <- Lyz1 > Lyz1_threshold & Serpinb6b < Serpinb6b_threshold

qplot(Lyz1, Serpinb6b, color = to_keep) +
  geom_vline(xintercept = Lyz1_threshold, linetype = 2) +
  geom_hline(yintercept = Serpinb6b_threshold, linetype = 2) +
  scale_color_brewer(palette = "Dark2") +
  labs(subtitle = "Non-cluster-disrupted: Serpinb6b > 2.5 and Lyz1 > 0")


# Let’s now subset the data appropriately:

sce_qc2 <- sce_qc[, to_keep]

# Finally, technical variation can have a large effect on single-cell RNA-seq data. Unfortunately
# we don’t know the experimental design, but one of the key signs of batch effects is large variation 
# in the number of genes expressed across cells (Hicks et al. (2017)).
# Let’s see how this affects the principal components of the data:

plotQC(sce_qc2, type = 'find', var = 'total_features', ntop = 2e3)


# We see this has a huge effect on the overall variation, contributing to the first principal component.
# We can remove this effect using the handy normaliseExprs function in scater:
cat("> NormaliseExprs\n")
m <- model.matrix(~ sce_qc2$total_features)
sce_qc2 <- normaliseExprs(sce_qc2, design = m)
exprs(sce_qc2) <- norm_exprs(sce_qc2)

# Let’s tidy up all the SCESets we have lying around before we’re ready for the PhenoPath analysis:
sce <- sce_qc2
rm(sce_qc, sce_qc2)
print(sce)

cat(paste0("... save data\n"))
outfile <- file.path(outFolder, "sce.Rdata")
save(sce, file=outfile)
cat(paste0("... written: ", outfile, "\n"))

cat("***** DONE\n")
cat(paste0(startTime, " - ", Sys.time(), "\n"))
cat(dim(sce))

# We applied ND to 820 cells using the 7,500 most vari-able genes, following the previously published linear
# analysis
