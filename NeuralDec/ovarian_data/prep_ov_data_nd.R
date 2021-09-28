# Rscript prep_ov_data_nd.R

nMostVar <- 1000

inFolder <- file.path("../../../ovarian_project/tcga_data/DOWNLOAD_TCGA_GTEX_OV_RECOUNT2/")
# raw_counts_dt <- get(load(file.path(inFolder, "all_counts_onlyPF_0.6.Rdata")))
tcga_annot_dt <- get(load(file.path(inFolder, "tcga_sampleAnnot.Rdata")))
gene_dt <- get(load(file.path(inFolder, "gene_dt.Rdata")))

#### prep annot

gene_dt$geneID <- gsub("(.+)\\..+", "\\1", gene_dt$geneID)
gene_dt <- unique(gene_dt)
stopifnot(!duplicated(gene_dt$geneID))
ens2symb <- setNames(gene_dt$geneSymb, gene_dt$geneID)

### prep count
filtNorm_dt <- get(load(file.path("../../../ovarian_project/phenopath/PHENOPATH_RECOUNT2_TCGA_OV/ov_data_gcNorm.Rdata")))
# Kaspar advice
# "I would expect log-transformed counts to work best "
log_filtNormCount <- log2(filtNorm_dt + 1)
stopifnot(!duplicated(tcga_annot_dt$cgc_sample_id))
rownames(tcga_annot_dt) <- tcga_annot_dt$cgc_sample_id
stopifnot(any(colnames(log_filtNormCount) %in% rownames(tcga_annot_dt)))
tcga_annot_dt <- tcga_annot_dt[rownames(tcga_annot_dt) %in% colnames(log_filtNormCount),]
stopifnot(rownames(tcga_annot_dt) %in% colnames(log_filtNormCount))

########## keep most variable
all_vars <- apply(log_filtNormCount, 1, var)
all_vars <- sort(all_vars, decreasing = TRUE)
mostVars <- names(all_vars)[1:nMostVar]

stopifnot(mostVars %in% rownames(log_filtNormCount))  
log_filtNormCount_mostVar <- log_filtNormCount[rownames(log_filtNormCount) %in% mostVars,]

########## prep out - mostVars
log_filtNormCount_mostVar <- as.data.frame(log_filtNormCount_mostVar)

samps <- colnames(log_filtNormCount_mostVar)

log_filtNormCount_mostVar$geneID_ensembl0 <- rownames(log_filtNormCount_mostVar)
log_filtNormCount_mostVar$geneID_ensembl <-  rownames(log_filtNormCount_mostVar)
stopifnot(!duplicated(log_filtNormCount_mostVar$geneID_ensembl))




stopifnot(log_filtNormCount_mostVar$geneID_ensembl0 %in% gene_dt$geneID)

log_filtNormCount_mostVar$geneID_symb <- ens2symb[paste0(log_filtNormCount_mostVar$geneID_ensembl0)]
stopifnot(!is.na(log_filtNormCount_mostVar$geneID_symb))

log_filtNormCount_mostVar <- log_filtNormCount_mostVar[,c("geneID_symb", "geneID_ensembl", samps)]


write.table(log_filtNormCount_mostVar, file = "log_filtNormCount_mostVar_dt.txt", sep=",", col.names=TRUE, row.names=FALSE, quote=F)


########## prep out - all
log_filtNormCount <- as.data.frame(log_filtNormCount)

samps <- colnames(log_filtNormCount)

log_filtNormCount$geneID_ensembl0 <- rownames(log_filtNormCount)
log_filtNormCount$geneID_ensembl <-  rownames(log_filtNormCount)
stopifnot(!duplicated(log_filtNormCount$geneID_ensembl))


gene_dt$geneID <- gsub("(.+)\\..+", "\\1", gene_dt$geneID)
gene_dt <- unique(gene_dt)

stopifnot(log_filtNormCount$geneID_ensembl0 %in% gene_dt$geneID)
stopifnot(!duplicated(gene_dt$geneID))
ens2symb <- setNames(gene_dt$geneSymb, gene_dt$geneID)

log_filtNormCount$geneID_symb <- ens2symb[paste0(log_filtNormCount$geneID_ensembl0)]
stopifnot(!is.na(log_filtNormCount$geneID_symb))

log_filtNormCount <- log_filtNormCount[,c("geneID_symb", "geneID_ensembl", samps)]


write.table(log_filtNormCount, file = "log_filtNormCount_dt.txt", sep=",", col.names=TRUE, row.names=FALSE, quote=F)


########## out annotation

keepcols <- c(
"cgc_case_clinical_stage",
"gdc_cases.demographic.year_of_birth",
"cgc_case_days_to_death",
"gdc_cases.diagnoses.days_to_birth",
"gdc_cases.diagnoses.age_at_diagnosis",
"cgc_slide_percent_normal_cells",
"cgc_slide_percent_neutrophil_infiltration",
"cgc_slide_percent_monocyte_infiltration",
"cgc_slide_percent_lymphocyte_infiltration",
"cgc_case_id")


tcga_annot_dt_out <- tcga_annot_dt[,keepcols] 
write.table(tcga_annot_dt_out, file = "tcga_annot_dt.txt", sep=",", col.names=TRUE, row.names=FALSE, quote=F)



