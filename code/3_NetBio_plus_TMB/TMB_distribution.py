## differentially expressed pathways
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import os, time
from collections import defaultdict
import lifelines; from lifelines.statistics import logrank_test; from lifelines import KaplanMeierFitter
exec(open('../utilities/parse_patient_data.py').read())
exec(open('../utilities/utilities_for_network_based_ML.py').read())


## Initialize
nGene = 200
qval = 0.01
study = 'Liu' # 'Liu', 'IMvigor210'
compare1 = ['TMB']
if study == 'IMvigor210':
	compare2 = ['TMB.plus.PD-L1_NetBio']
if study == 'Liu':
	compare2 = ['TMB.plus.PD1_NetBio']

target_dic = {'IMvigor210':'PD-L1', 'Liu':'PD1'}



# data directory
data_dir = '../../result/3_NetBio_plus_TMB'

# output directory
fo_dir = data_dir
if os.path.isdir('%s/TMB_distribution'%(fo_dir)) == False:
	os.mkdir('%s/TMB_distribution'%fo_dir)
fo_dir = '%s/TMB_distribution'%fo_dir


# output stats
output = defaultdict(list)


# import data and check TMB distribution
for fi in os.listdir(data_dir):
	if 'PredictedResponse.txt' in fi:
		df = pd.read_csv('%s/%s'%(data_dir, fi), sep='\t')
		df = df.loc[df['nGene']==nGene,:].loc[df['qval']==qval,:]
		# study
		try:
			_, pdf = parse_immunotherapy_survival(study)
			TMB = parse_clinical_features(study, clinical_feature='TMB')
		except: continue
		# test start
		print('testing %s, %s'%(study, time.ctime()))

		# merged
		merged = pd.merge(df, pdf, on='sample', how='inner')
		
		
		## os plot for original VS reclassified group
		original_dic = defaultdict(list)
		reclassified_dic = defaultdict(list)
		tmp1 = merged.loc[merged['study']==study,:].loc[merged['nGene']==nGene,:].loc[merged['qval']==qval,:].loc[merged['test_type'].isin(compare1),:]
		tmp2 = merged.loc[merged['study']==study,:].loc[merged['nGene']==nGene,:].loc[merged['qval']==qval,:].loc[merged['test_type'].isin(compare2),:]
		for sample, response1 in zip(tmp1['sample'].tolist(), tmp1['predicted_response']):
			response2 = tmp2.loc[tmp2['sample']==sample,:]['predicted_response'].tolist()[0]
			if response1 == 1:
				R1 = 'R'
			else:
				R1 = 'NR'
			if response2 == 1:
				R2 = 'R'
			else:
				R2 = 'NR'
			original_dic[R1].append(sample)
			if R1 != R2:
				reclassified_dic['%s2%s'%(R1, R2)].append(sample)
		# samples
		stat_df = defaultdict(list)

		original_R, original_NR = original_dic['R'], original_dic['NR']
		R2NR, NR2R = reclassified_dic['R2NR'], reclassified_dic['NR2R']
		original_R_exc, original_NR_exc = list(set(original_R)-set(R2NR)), list(set(original_NR)-set(NR2R))

			
		scores = []; xlabels = ['R2R', 'R2NR', 'NR2NR', 'NR2R']
		for sList in [original_R_exc, R2NR, original_NR_exc, NR2R]:
			scores.append(TMB.loc[TMB['sample'].isin(sList),:]['status'].tolist())
		

		output = defaultdict(list)
		
		for s1_idx, s1 in enumerate([original_R, R2NR, original_NR, NR2R]):
			for s2_idx, s2 in enumerate([original_R, R2NR, original_NR, NR2R]):
				if s1_idx < s2_idx:
					score1 = TMB.loc[TMB['sample'].isin(s1),:]['status'].tolist()
					score2 = TMB.loc[TMB['sample'].isin(s2),:]['status'].tolist()
					_, mwu_pval = stat.mannwhitneyu(score1, score2)
					output['study'].append(study)
					output['group1'].append(xlabels[s1_idx])
					output['group2'].append(xlabels[s2_idx])
					output['mwu_pval'].append(mwu_pval)
		output = pd.DataFrame(output)
		output.to_csv('%s/%s_stat.txt'%(fo_dir, study), sep='\t')



		plt.boxplot(scores)
		plt.ylabel('Tumor Mutational Burden')
		plt.xticks([1,2,3,4], xlabels, rotation=90)
		plt.tight_layout()
		plt.savefig('%s/%s.jpg'%(fo_dir, study), format='jpg')
		plt.savefig('%s/%s.eps'%(fo_dir, study), format='eps', dpi=300)
		plt.close()		
