## compilation of TCGA molecular subtypes
## using "TCGAbiolinks"
## https://www.bioconductor.org/packages/devel/bioc/vignettes/TCGAbiolinks/inst/doc/subtypes.html

## set directory
setwd('E:/old_Jurgen/research_result/BLCA_CISPLATIN_results/github_NetBio.ML/code/5_from_ICI_to_TCGA')

## download TCGAbiolinks package
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("TCGAbiolinks")

## subtypes
subtypes <- PanCancerAtlas_subtypes()

