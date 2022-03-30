##---------------------------------------------------
## TMM Normalization of RSEM expected counts
## TCGA patients
##---------------------------------------------------

## Import library
library(edgeR)
library(tibble)

## Initialize
#tcga_list = c('TCGA-BLCA')
tcga_list = c('TCGA-BRCA','TCGA-GBM','TCGA-OV','TCGA-LUAD','TCGA-UCEC','TCGA-KIRC',
              'TCGA-HNSC','TCGA-LGG','TCGA-THCA','TCGA-LUSC','TCGA-PRAD','TCGA-SKCM',
              'TCGA-COAD','TCGA-STAD','TCGA-BLCA','TCGA-LIHC','TCGA-CESC','TCGA-KIRP',
              'TCGA-SARC','TCGA-LAML','TCGA-ESCA','TCGA-PAAD','TCGA-PCPG','TCGA-READ',
              'TCGA-TGCT','TCGA-THYM','TCGA-KICH','TCGA-ACC','TCGA-MESO','TCGA-UVM',
              'TCGA-DLBC','TCGA-UCS','TCGA-CHOL')
fi_dir = 'D:/research_result/cGAN/data/TCGAbiolinks'
setwd(fi_dir)


## TMM normalization for each cancer types
for (cancer in tcga_list){
  # set output directory
  tmp_fo_dir = paste0(fi_dir, '/GDCdata/', cancer)
  setwd(tmp_fo_dir)
  
  print(paste0('testing ', cancer, ' / ', Sys.time()))
  
  if (file.exists('rna_seq_table.csv')==TRUE){
    if (file.exists('TMM_rna_seq.txt')==FALSE){
      # Import expression data (HTseq count)
      count = read.csv('rna_seq_table.csv', header=TRUE, check.names=FALSE)
      rownames(count) = NULL
      colnames(count) = c(names(count))
      colnames(count)[1] = 'Gene stable ID'

      # ---------------
      # Compute TMM
      tmp <- count
      tmp$`Gene stable ID` <- NULL
      
      # Create DGEList object
      dgList <- DGEList(counts=tmp, genes=count$`Gene stable ID`)
      #dgList$samples
      #head(dgList$counts)
      #head(dgList$genes)
      
      # Filtering
      countsPerMillion <- cpm(dgList)
      #summary(countsPerMillion)
      countCheck <- countsPerMillion > 1
      
      keep<- which(rowSums(countCheck) >= 2)
      dgList <- dgList[keep,]

      
      # Normalization (TMM)
      dgList <- calcNormFactors(dgList, method='TMM')
      cps <- cpm(dgList) #, normalized.lib.sizes = TRUE)
      output <- data.frame(dgList$genes, cps, check.names=FALSE)
      
      cps2 <- cpm(dgList, log=TRUE, prior.count=1) #, normalized.lib.sizes = TRUE)
      output2 <- data.frame(dgList$genes, cps2, check.names=FALSE)
      
      # Return normalized count results
      write.table(output, file='TMM_rna_seq.txt', sep='\t',quote = FALSE, row.names=FALSE)  
      write.table(output2, file='TMM_log2_rna_seq.txt', sep='\t',quote = FALSE, row.names=FALSE)  
      
    }
    
  }
  
  # set directory to parent folder
  setwd(fi_dir)
}
