## TMB calculation method from "Correlate tumor mutation burden with immune signatures in human cancers", BMC Immunology, 2019
## TMB = total number of truncating mutations * 2.0 + total number of non-truncating mutations * 1.0
## truncating mutations : Nonsense, frame-shift deletion or insertion, splice-site mutations
## non-truncating mutations : missense, in-frame deletion or insertion, nonstop mutations 
import os, time
import pandas as pd
from collections import defaultdict


## truncating and non-truncating mutations
truncating_mutations = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site']
non_truncating_mutations = ['Missense_Mutation', 'In_Frame_Ins', 'In_Frame_Del', 'Nonstop_Mutation']



## calculate TMB 
fldr_list = os.listdir('/home/user/shared/TCGAbiolinks/GDCdata')
for fldr in fldr_list:
	if 'TCGA' in fldr:
		print('testing %s, %s'%(fldr, time.ctime()))





		fiList = os.listdir('/home/user/shared/TCGAbiolinks/GDCdata/%s'%fldr)
		for fi in fiList:
			if ('SNV' in fi) and (not 'TMB' in fi):
				output = defaultdict(list)
				output_col = ['Tumor_Sample_Barcode', 'TMB']
				# load data
				df = pd.read_csv('/home/user/shared/TCGAbiolinks/GDCdata/%s/%s'%(fldr, fi), low_memory=False)
				# sample IDs
				samples = list(set(df['Tumor_Sample_Barcode'].tolist()))
				# TMB per patient
				for sample in samples:
					tmp = df.loc[df['Tumor_Sample_Barcode']==sample,:]
					num_truncating = len(tmp.loc[tmp['Variant_Classification'].isin(truncating_mutations),:])
					num_nontruncating = len(tmp.loc[tmp['Variant_Classification'].isin(non_truncating_mutations),:])
					TMB = num_truncating * 2.0 + num_nontruncating * 1.0
					output['Tumor_Sample_Barcode'].append(sample)
					output['TMB'].append(TMB)
				output = pd.DataFrame(data=output, columns=output_col)
				output.to_csv('/home/user/shared/TCGAbiolinks/GDCdata/%s/%s_TMB.txt'%(fldr, fi.replace('.csv', '')), sep='\t', index=False)



## complete!!!
print('process complete, %s'%time.ctime())
