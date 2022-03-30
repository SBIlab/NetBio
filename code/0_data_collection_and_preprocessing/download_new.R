library(TCGAbiolinks)
library(SummarizedExperiment)

setwd('E:/docu/master/Documents/TCGAbiolinks')

tcga_list = c('TCGA-BRCA','TCGA-GBM','TCGA-OV','TCGA-LUAD','TCGA-UCEC','TCGA-KIRC',
              'TCGA-HNSC','TCGA-LGG','TCGA-THCA','TCGA-LUSC','TCGA-PRAD','TCGA-SKCM',
              'TCGA-COAD','TCGA-STAD','TCGA-BLCA','TCGA-LIHC','TCGA-CESC','TCGA-KIRP',
              'TCGA-SARC','TCGA-LAML','TCGA-ESCA','TCGA-PAAD','TCGA-PCPG','TCGA-READ',
              'TCGA-TGCT','TCGA-THYM','TCGA-KICH','TCGA-ACC','TCGA-MESO','TCGA-UVM',
              'TCGA-DLBC','TCGA-UCS','TCGA-CHOL')

for(i in tcga_list){
  ### expression ###
  query <- GDCquery(project = i,
                    data.category = 'Transcriptome Profiling',
                    data.type = 'Gene Expression Quantification',
                    workflow.type = 'HTSeq - Counts',
                    sample.type = 'Primary Tumor')
  if(is.null(query) != TRUE){
    GDCdownload(query)
    data <- GDCprepare(query)
    eset <- assay(data)
    write.csv(eset, file = paste0('E:/docu/master/Documents/TCGAbiolinks/GDCdata/',i,'/rna_seq_table.csv'))
  }
  
  ### simple nucleotide variation ###
  for(j in c('muse','mutect2','somaticsniper','varscan2')){
    maf <- GDCquery_Maf(strsplit(i, '-')[[1]][2], pipelines = j)
    write.csv(maf, file = paste0('E:/docu/master/Documents/TCGAbiolinks/GDCdata/',i,'/SNV_',j,'.csv'))
  }
  
  ### Copy Number Variation ###
  query <- GDCquery(project = i,
                    data.category = 'Copy Number Variation',
                    data.type = 'Copy Number Segment',
                    sample.type = 'Primary Tumor')
  if(is.null(query) != TRUE){
    GDCdownload(query)
    data <- GDCprepare(query)
    eset <- assay(data)
    write.csv(eset, file = paste0('E:/docu/master/Documents/TCGAbiolinks/GDCdata/',i,'/CNV_table.csv'))
  }
  
  ### clinical ###
  ## drug response ##
  query <- GDCquery(project = i,
                    data.category = 'Clinical',
                    data.type = 'Clinical Supplement',
                    data.format = 'BCR Biotab')
  if(is.null(query) != TRUE){
    GDCdownload(query)
    data <- GDCprepare(query)
    eset <- data[[paste0('clinical_drug_',tolower(strsplit(i,'-')[[1]][2]))]]
    write.csv(eset, file = paste0('E:/docu/master/Documents/TCGAbiolinks/GDCdata/',i,'/clinical_drug_table.csv'))
  }

  ## patient information ##
  query <- GDCquery(project = i,
                    data.category = 'Clinical',
                    data.type = 'Clinical Supplement',
                    data.format = 'BCR Biotab')
  if(is.null(query) != TRUE){
    GDCdownload(query)
    data <- GDCprepare(query)
    eset <- data[[paste0('clinical_patient_',tolower(strsplit(i,'-')[[1]][2]))]]
    write.csv(eset, file = paste0('E:/docu/master/Documents/TCGAbiolinks/GDCdata/',i,'/clinical_patient_table.csv'))
  }
  
}