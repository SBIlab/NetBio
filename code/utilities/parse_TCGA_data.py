## parse TCGA data
import pandas as pd
from collections import defaultdict
import numpy as np
import scipy.stats as stat
import os, time

def TCGA_ssGSEA(cancer_type, parse_reactome=True, simplify_barcode=True):
	'''
	Input
	cancer_type: 'BLCA', 'SKCM' (melanoma), 'STAD' (gastric cancer)
	simplify_barcode: if True, duplicate samples are removed
	'''
	fi_dir = '../../data/TCGA/ssgsea'
	df = pd.DataFrame()
	if 'TCGA-%s'%cancer_type in os.listdir(fi_dir):
		df = pd.read_csv('%s/TCGA-%s/ssgsea.txt'%(fi_dir, cancer_type), sep='\t')
		df = df.rename(columns={'testType':'pathway'})
		df_col = ['pathway']
		for sample in df.columns[1:]:
			if 'TCGA' in sample:
				df_col.append(sample)
		df = pd.DataFrame(data=df, columns=df_col)

		if parse_reactome == True:
			df = df.loc[df['pathway'].str.contains('REACTOME'),:]
			if simplify_barcode == True:
				rename_dic = {}
				cols = ['pathway']
				samples = []
				for col in df.columns[1:]:
					samples.append(col[:12])
				for sample, col in zip(samples, df.columns[1:]):
					if samples.count(sample) == 1:
						rename_dic[col] = sample
						cols.append(sample)
				df = df.rename(columns=rename_dic)
				df = pd.DataFrame(data=df, columns=cols)
	return df


def TCGA_gene_expression(cancer_type, official_gene_symbol=True, simplify_barcode=True):
	'''
	Input
	cancer_type: 'BLCA', 'SKCM' (melanoma), 'STAD' (gastric cancer)
	simplify_barcode: if True, duplicate samples are removed
	'''
	fi_dir = '../../data/TCGA'
	df = pd.DataFrame()
	if 'TCGA-%s'%cancer_type in os.listdir(fi_dir):
		df = pd.read_csv('%s/TCGA-%s/TMM_rna_seq.txt'%(fi_dir, cancer_type), sep='\t')
		if official_gene_symbol == True:
			edf = pd.read_csv('/home/user/shared/TCGAbiolinks/ENSG_GENESYMBOL.txt', sep='\t')
			edf = edf.rename(columns={'Gene name':'genes'})
			df_col = df.columns
			df = df.rename(columns={'genes':'Gene stable ID'})
			df = pd.merge(edf, df, on='Gene stable ID', how='inner')
			
			tmp_df = pd.DataFrame(data=df, columns=df_col).sort_values(by='genes')
			df = defaultdict(list)
			geneList1, geneList2 = [], []
			for gene in list(set(tmp_df['genes'].tolist())):
				if list(tmp_df['genes'].tolist()).count(gene) == 1:
					geneList1.append(gene)
				else:
					geneList2.append(gene)

			df = tmp_df.loc[tmp_df['genes'].isin(geneList1),:]
	
			for gene in list(set(geneList2)):
				tmp2 = defaultdict(list)
				tmp2['genes'].append(gene)
				for sample in df.columns[1:]:
					exp = np.median(tmp_df.loc[tmp_df['genes']==gene,:][sample].tolist())
					tmp2[sample].append(exp)
				tmp2 = pd.DataFrame(data=tmp2, columns=tmp_df.columns)
				df = df.append(tmp2)
			df = df.sort_values(by='genes')

			if simplify_barcode == True:
				rename_dic = {}
				cols = ['genes']
				samples = []
				for col in df.columns[1:]:
					samples.append(col[:12])
				for sample, col in zip(samples, df.columns[1:]):
					if samples.count(sample) == 1:
						rename_dic[col] = sample
						cols.append(sample)
				df = df.rename(columns=rename_dic)
				df = pd.DataFrame(data=df, columns=cols)
	return df


def TCGA_TMB(cancer_type, mut_data='mutect2', simplify_barcode=True):
	'''
	Input
	cancer_type: 'BLCA', 'SKCM' (melanoma), 'STAD' (gastric cancer)
	mut_data: 'mutect2' (default), 'muse', 'somaticsniper', 'varscan2'
	simplify_barcode: if True, duplicate samples are removed
	'''
	fi_dir = '../../data/TCGA'
	df = pd.DataFrame()
	if 'TCGA-%s'%cancer_type in os.listdir(fi_dir):
		df = pd.read_csv('%s/TCGA-%s/SNV_%s_TMB.txt'%(fi_dir, cancer_type, mut_data), sep='\t')
		df = df.rename(columns={'Tumor_Sample_Barcode':'sample', 'TMB':'status'})
		if simplify_barcode == True:
			output = defaultdict(list)
			samples = []
			for sample in df['sample'].tolist():
				samples.append(sample[:12])
			for sample, TMB_ in zip(samples, df['status'].tolist()):
				if samples.count(sample) == 1:
					output['sample'].append(sample)
					output['status'].append(TMB_)
			df = pd.DataFrame(data=output, columns=['sample', 'status'])

	return df


def TCGA_immune_landscape(data_type):
	'''
	Input
	data_type : 'homologous_repair_deficiency' (or 'HRD'), 'CIBERSORT', 'leukocyte', 'MHC_SNV', 'indel_neoantigen_counts'
	'''
	fi_dir = '../../data/TCGA/immune_landscape'
	if (data_type == 'homologous_repair_deficienty') or (data_type == 'HRD'):
		fiName = 'TCGA.HRD_withSampleID.txt'
	elif data_type == 'CIBERSORT':
		fiName = 'TCGA.Kallisto.fullIDs.cibersort.relative.tsv'
	elif data_type == 'leukocyte':
		fiName = 'TCGA_all_leuk_estimate.masked.20170107.tsv'
	elif data_type == 'MHC_SNV':
		fiName = 'TCGA_pMHC_SNV_sampleSummary_MC3_v0.2.8.CONTROLLED_170404.tsv'
	elif data_type == 'indel_neoantigen_counts':
		fiName = 'TCGA_PCA.mc3.v0.2.8.CONTROLLED.filtered.sample_neoantigens_10062017.tsv'
	df = pd.read_csv('%s/%s'%(fi_dir, fiName), sep='\t')
	col_dic = {}
	col_list = ['sampleID', 'SampleID', 'sample', 'barcode']
	for col in col_list:
		col_dic[col] = 'sampleID'
	df = df.rename(columns=col_dic)
	df = df.rename(columns={'CancerType':'cancer_type'})
	# 
	sampleList = []
	for sample in df['sampleID'].tolist():
		sample = sample.replace('.','-')[:12]
		sampleList.append(sample)
	df['sample']=sampleList
	return df	

