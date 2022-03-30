## parse data for
# 1. gene expression
# 2. reactome pathway expression
# 3. immunotherapy response
# 4. immunotherapy response survival


import pandas as pd
from collections import defaultdict
import scipy.stats as stat
import numpy as np
import time, os



## pathway expression and immunotherapy response
def parse_reactomeExpression_and_immunotherapyResponse(dataset, method='ssgsea', drug_treatment='pre', Prat_cancer_type='MELANOMA'):
	'''
	Input
	dataset : 'IMvigor210', 'Liu', 'Riaz', 'Gide', 'Prat', 'Kim', 'Auslander'
	method : 'ssgsea'
	drug_treatment : 'pre', 'on', 'all'
	Prat_cancer_type : 'HEADNECK', 'ADENO', 'SQUAMOUS', 'MELANOMA'
	'''
	edf = parse_gene_expression(dataset)
	epdf = parse_reactome_expression(dataset)
	pdf = parse_immunotherapy_response(dataset)
	# features, labels
	exp_dic, responses = defaultdict(list), []
	e_samples = []
	for sample, response in zip(pdf['Patient'].tolist(), pdf['Response'].tolist()):
		# labels
		binary_response = response
		if 'NR' == response:
			binary_response = 0
		if 'R' == response:
			binary_response = 1
		# features
		for e_sample in epdf.columns:
			tmp = []
			if (sample in e_sample) or (e_sample in sample):
				if 'Riaz' in dataset:
					if drug_treatment == 'all':
						if not '%s_'%sample in e_sample:
							continue
					if drug_treatment == 'on':
						if not '%s_On_'%sample in e_sample:
							continue
					if drug_treatment == 'pre':
						if not '%s_Pre_'%sample in e_sample:
							continue

				elif 'Huang' in dataset:
					if sample != e_sample: 
						continue
					if drug_treatment == 'pre':
						if 'Post' in str(pdf.loc[pdf['Patient']==sample,:]['drug_treatment_info'].tolist()[0]):
							continue
					if drug_treatment == 'post':
						if 'Pre' in str(pdf.loc[pdf['Patient']==sample,:]['drug_treatment_info'].tolist()[0]):
							continue
						
				elif 'Prat' in dataset:
					if not 'all' == Prat_cancer_type.lower():						
						new_samples = pdf.loc[pdf['CANCER2']==Prat_cancer_type,:]['Patient'].tolist()
						if (not e_sample in new_samples) or (not sample in new_samples):
							continue
					else:
						if not sample == e_sample:
							continue
				else:
					if not sample == e_sample:
						continue
				e_samples.append(e_sample)
				responses.append(binary_response)	
	edf = pd.DataFrame(data=edf, columns=np.append(['genes'], e_samples))
	epdf = pd.DataFrame(data=epdf, columns=np.append(['pathway'], e_samples))
	responses = np.array(responses)
	return e_samples, edf, epdf, responses





# pathway expression
def parse_reactome_expression(dataset, method='ssgsea'):
	'''
	Input
	dataset : 'IMvigor210', 'Liu', 'Riaz', 'Gide', 'Prat', 'Kim', 'Auslander'
	method : 'ssgsea'
	'''
	epdf = pd.DataFrame()
	# directory
	data_dir = '../../data/ImmunoTherapy'
	fldr = dataset
	for tmp_fldr in os.listdir(data_dir):
		if (dataset in tmp_fldr) and (not '.txt' in tmp_fldr) and (not '.R' in tmp_fldr):
			fldr = tmp_fldr
			break
	# import data
	try:
		epdf = pd.read_csv('%s/%s/pathway_expression_ssgsea.txt'%(data_dir, fldr), sep='\t')
	except: 
		if 'IMvigor210' in fldr:
			epdf = pd.read_csv('%s/%s/REACTOME_ssgsea.txt'%(data_dir, fldr), sep='\t')
			epdf = epdf.rename(columns={'testType':'pathway'})
		elif ('Liu' in fldr) or ('Gide' in fldr):
			epdf = pd.read_csv('%s/%s/ssgsea.txt'%(data_dir, fldr), sep='\t')
		elif len(epdf) == 0:
			if dataset in os.listdir('%s/ssgsea'%data_dir):
				tmp_dir = '%s/ssgsea/%s/msigdb_c2'%(data_dir, dataset)
				if 'ssgsea.txt' in os.listdir(tmp_dir):
					epdf = pd.read_csv('%s/ssgsea.txt'%tmp_dir, sep='\t')
	# data cleanup
	if len(epdf)>0:
		epdf = epdf.rename(columns={'testType':'pathway'})
		epdf = epdf.loc[epdf['pathway'].str.contains('REACTOME_'),:]
		epdf = epdf.dropna()
	return epdf



## immunotherapy response
def parse_immunotherapy_response(dataset):
	'''
	Input
	dataset : 'IMvigor210', 'Liu', 'Riaz', 'Gide', 'Prat', 'Kim', 'Auslander'
	'''
	# directory
	data_dir = '../../data/ImmunoTherapy'
	fldr = dataset
	for tmp_fldr in os.listdir(data_dir):
		if (dataset in tmp_fldr) and (not '.txt' in tmp_fldr) and (not '.R' in tmp_fldr):
			fldr = tmp_fldr
			break
	# import data
	pdf = pd.read_csv('%s/%s/patient_df.txt'%(data_dir, fldr), sep='\t')
	if ('Liu' in fldr) or ('Gide' in fldr):
		pdf['Response'] = pdf['Response'].astype(str)
		pdf = pdf.loc[pdf['Response'].isin(['PD', 'PR', 'CR', 'SD']),:]
		tmp_response = []
		for r in pdf['Response'].tolist():
			if r in ['CR', 'PR']:
				tmp_response.append('R')
			if r in ['SD', 'PD']:
				tmp_response.append('NR')
		pdf['Response'] = tmp_response
	return pdf


## gene expression 
def parse_gene_expression(dataset):
	'''
	Input
	dataset : 'IMvigor210', 'Liu', 'Riaz', 'Gide', 'Prat', 'Kim', 'Auslander'
	'''
	# directory
	data_dir = '../../data/ImmunoTherapy'
	fldr = dataset
	for tmp_fldr in os.listdir(data_dir):
		if (dataset in tmp_fldr) and (not '.txt' in tmp_fldr) and (not '.R' in tmp_fldr):
			fldr = tmp_fldr
			break
	# import data
	edf = pd.DataFrame()
	if 'IMvigor210' in fldr:
		edf = pd.read_csv('%s/%s/expression_geneSymbol.txt'%(data_dir, fldr), sep='\t')
		edf = edf.rename(columns={'gene_id':'genes'})
	else:
		try:
			edf = pd.read_csv('%s/%s/TMM_rna_seq.txt'%(data_dir, fldr), sep='\t')

		except: 
			try:
				edf = pd.read_csv('%s/%s/expression_mRNA.norm3.txt'%(data_dir, fldr), sep='\t')
				edf = edf.rename(columns={'gene_id':'genes'})
			except: pass
	edf = edf.dropna()
	return edf


## immunotherapy response overall survival
def parse_immunotherapy_survival(dataset, survival_type = 'os'):
	'''
	Input
	dataset : IMvigor210 (phenotype_final.txt), Liu (clinical_original.txt), Riaz (Patient_clinical.txt), Gide (clinical.txt')
	survival_type : os (overall survival), pfs (progression free survival)
	'''
	output = defaultdict(list)
	# directory
	data_dir = '../../data/ImmunoTherapy'
	fldr = dataset
	for tmp_fldr in os.listdir(data_dir):
		if (dataset in tmp_fldr) and (not '.txt' in tmp_fldr) and (not '.R' in tmp_fldr):
			fldr = tmp_fldr
			break
	
	# import data
	if 'Riaz' in dataset:
		pdf = pd.read_csv('%s/%s/Patient_clinical.txt'%(data_dir, fldr), sep='\t')
	if 'Liu' in dataset:
		pdf = pd.read_csv('%s/%s/clinical_original.txt'%(data_dir, fldr), sep='\t', encoding='cp949')
	if 'IMvigor210' in dataset:
		pdf = pd.read_csv('%s/%s/phenotype_final.txt'%(data_dir, fldr), sep='\t')
	if 'Gide' in dataset:
		pdf = pd.read_csv('%s/%s/clinical.txt'%(data_dir, fldr), sep='\t')
	
	# columns
	if survival_type.lower() == 'os':
		dataset_list = ['IMvigor210', 'Liu', 'Gide']
		sample_col = ['sampleID', 'samples', 'ID']
		os_col = ['survival_month', 'OS', 'surv_dt.time']
		os_status_col = ['survival_status', 'dead', 'surv_dt.status']
		os_type = ['Months', 'Days', 'Days']
		
		# index
		idx = dataset_list.index(dataset)
		sc, oc, osc, ot = sample_col[idx], os_col[idx], os_status_col[idx], os_type[idx]

		# parse data
		for sample, OS, os_status in zip(pdf[sc].tolist(), pdf[oc].tolist(), pdf[osc].tolist()):
			if (str(os_status) == 'nan') or (str(OS) == 'nan'):
				continue
			# os status
			if ('True'==str(os_status)) or ('DEAD'==str(os_status).upper()) or ('DECEASED'==str(os_status).upper()):
				os_status = 1
			if ('False'==str(os_status)) or ('ALIVE'==str(os_status).upper()):
				os_status = 0

			output['sample'].append(sample)
			output['os'].append(float(OS))
			output['os_status'].append(int(os_status))
			output['os_type'].append(ot)
		output = pd.DataFrame(data=output, columns=['sample', 'os', 'os_status', 'os_type'])
	
	if survival_type.lower() == 'pfs':
		dataset_list = ['Liu', 'Gide']
		sample_col = ['samples', 'ID']
		pfs_col = ['PFS', 'surv_dt2.time']
		pfs_status_col = ['progressed', 'surv_dt2.status']
		pfs_type = ['Days', 'Days']
		
		# index
		idx = dataset_list.index(dataset)
		sc, pc, psc, pt = sample_col[idx], pfs_col[idx], pfs_status_col[idx], pfs_type[idx]

		# parse data
		for sample, PFS, pfs_status in zip(pdf[sc].tolist(), pdf[pc].tolist(), pdf[psc].tolist()):
			if (str(pfs_status) == 'nan') or (str(PFS) == 'nan'):
				continue
			# os status
			if ('True'==str(pfs_status)) or ('DEAD'==str(pfs_status).upper()) or ('DECEASED'==str(pfs_status).upper()):
				os_status = 1
			if ('False'==str(pfs_status)) or ('ALIVE'==str(pfs_status).upper()):
				os_status = 0

			output['sample'].append(sample)
			output['pfs'].append(float(PFS))
			output['pfs_status'].append(int(pfs_status))
			output['pfs_type'].append(pt)
		output = pd.DataFrame(data=output, columns=['sample', 'pfs', 'pfs_status', 'pfs_type'])
	return pdf, output



## parse other clinical features
def parse_clinical_features(dataset, clinical_feature='TMB'):
	'''
	Input
	dataset : IMvigor210
	clinical_feature : TMB, immune_phenotype, IC (PD-L1 expression on immune cells), TC (PD-L1 expression on tumor cells)
	---
	Output columns
	sample 
	status
	'''
	output = defaultdict(list)
	# columns
	cDic = { 'IMvigor210': {'TMB':'FMOne mutation burden per MB', 'immune_phenotype':'Immune phenotype', 'IC':'IC Level', 'TC':'TC Level'},
			'Liu' : {'TMB':'nonsyn_muts'}}
	try:
		col = cDic[dataset][clinical_feature]
	except:
		print('wrong input')
		raise ValueError


	# directory
	data_dir = '../../data/ImmunoTherapy'
	fldr = dataset
	for tmp_fldr in os.listdir(data_dir):
		if (dataset in tmp_fldr) and (not '.txt' in tmp_fldr) and (not '.R' in tmp_fldr):
			fldr = tmp_fldr
			break
	# import data
	if 'IMvigor210' in dataset:
		pdf = pd.read_csv('%s/%s/pData.txt'%(data_dir, fldr))
		pdf = pdf.rename(columns={'sample_id':'sample'})
	if 'Liu' in dataset:
		pdf = pd.read_csv('%s/%s/clinical_original.txt'%(data_dir, fldr), sep='\t')
		pdf = pdf.rename(columns={'samples':'sample'})
	# parse data
	pdf = pdf.dropna(subset=['sample', col])
	for sample, status in zip(pdf['sample'].tolist(), pdf[col].tolist()):
		output['sample'].append(sample)
		output['status'].append(status)
	output = pd.DataFrame(data=output, columns=['sample', 'status'])
	# TC, IC
	if (clinical_feature == 'IC') or (clinical_feature == 'TC'):
		tmpdf = defaultdict(list)
		subgroups = sorted(pdf[col].value_counts().index.tolist())
		for sample, status in zip(pdf['sample'].tolist(), pdf[col].tolist()):
			tmpdf['sample'].append(sample)
			tmp = np.zeros(len(subgroups))
			idx = subgroups.index(status)
			tmp[idx] = 1
			for subgroup, idx in zip(subgroups, tmp):				
				tmpdf['%s_%s'%(clinical_feature, subgroup)].append(idx)
		output = pd.merge(output, pd.DataFrame(tmpdf), on='sample', how='inner')
	return output
		


## parse CIBERSORT results
def parse_CIBERSORT(dataset):
	'''
	Input : IMvigor210
	'''
	if 'IMvigor210' == dataset:
		df = pd.read_csv('/home/user/shared/ImmunoTherapy/IMvigor210/IMvigor210_cibersortx.txt', sep='\t')
	columns = []
	for col in df.columns:
		if ('P-value' == col) or ('Correlation' == col) or ('RMSE' == col):
			continue
		columns.append(col)
	df = pd.DataFrame(data=df, columns=columns)
	return df
