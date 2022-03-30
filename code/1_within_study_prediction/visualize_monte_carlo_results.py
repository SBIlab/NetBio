## visualize monte carlo prediction results
## 1. summarized prediction results
## 2. boxplot displaying monte carlo cross-validation prediction results
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
from collections import defaultdict
import pandas as pd
from operator import add



## Initialize
pval_cutoff = 0.05
fi_dir = '../../result/1_within_study_prediction'
test_size = 0.2
ML = 'LogisticRegression'
cohort_targets = {'PD1':['Liu', 'Riaz_pre', 'Kim'],
	'PD1_CTLA4':['Gide'],
	'PD-L1':['IMvigor210']}

controls = ['PD1', 'PD-L1', 'CTLA4', 'PD1_PD-L1_CTLA4', 'CD8T1', 'T_exhaust_Pos', 'CAF1', 'TAM_M2_M1_Pos', 'all-TME-Bio']
metrics = ['accuracy', 'F1'] # ['accuracy', 'precision', 'recall', 'F1']
fo_dir = fi_dir



## Load data
df = pd.read_csv('%s/monteCarlo_testsize_%s_%s.txt'%(fi_dir, test_size, ML), sep='\t')




## visualize
output = defaultdict(list)
output_col = ['study', 'metric', 'cohort_target', 'control', 'NetBio_mean', 'control_mean', 'ttest_pvalue', 'prediction_result']

for metric in metrics:
	summ_dic = defaultdict(list)
	studies = []
	summ_counts = []

	for study in df['study'].value_counts().index:
		studies.append(study)
		for targets in list(cohort_targets.keys()):
			if study in cohort_targets[targets]:
				drug_targets = targets
				break
		# NetBio
		NetBio = df.loc[df['study']==study,:].loc[df['test_size']==test_size,:].loc[df['test_type']=='NetBio',:].loc[df['ML']==ML,:][metric].tolist()

		# controls
		c_scores = [] # [[ scores 1 ], [ scores 2 ], ... ] // list of scores
		for control in controls:
			scores = df.loc[df['study']==study,:].loc[df['test_size']==test_size,:].loc[df['test_type']==control,:].loc[df['ML']==ML,:][metric].tolist()
			c_scores.append(scores)

		## 1. summarized prediction results
		summ = {'better':0, 'equal':0, 'worse':0}
		for control, scores in zip(controls, c_scores):
			_, pval = stat.ttest_ind(NetBio, scores)
			for key, value in zip(['study', 'metric', 'cohort_target', 'control', 'NetBio_mean', 'control_mean'], [study, metric, drug_targets, control, np.mean(NetBio), np.mean(scores)]):
				output[key].append(value)
			output['ttest_pvalue'].append(pval)
			if pval <= pval_cutoff:
				if np.mean(NetBio) > np.mean(scores):
					summ['better']+=1
					output['prediction_result'].append('better')
				else:
					summ['worse']+=1
					output['prediction_result'].append('worse')
			else:
				summ['equal']+=1
				output['prediction_result'].append('equal')
		for test_type in ['better', 'equal', 'worse']:
			percent_score = 100*summ[test_type]/len(c_scores)
			summ_dic[test_type].append(percent_score)
			summ_counts.append('%s_%s_%s'%(study, test_type, summ[test_type]))


		## 2. boxplot displaying monte carlo cross-validation prediction results
		all_scores = np.append([NetBio], c_scores, axis=0)
		labels = np.append(['NetBio'], controls)
		plt.close()
		plt.boxplot(list(all_scores))
		plt.xticks(np.arange(1,len(labels)+1), labels, rotation=90)
		plt.ylabel(metric)
		plt.xlabel('Features')
		plt.tight_layout()
		plt.savefig('%s/boxplots/%s_%s_%s_test_size_%s.jpg'%(fo_dir, study, metric, ML, test_size), format='jpg')
		plt.savefig('%s/boxplots/%s_%s_%s_test_size_%s.eps'%(fo_dir, study, metric, ML, test_size), format='eps', dpi=300)
		plt.close()


	## visualize #1
	plt.close()
	fig, ax = plt.subplots()
	ax.bar(studies, summ_dic['better'], label='better')
	ax.bar(studies, summ_dic['equal'], bottom=summ_dic['better'], label='equal')
	ax.bar(studies, summ_dic['worse'], bottom=list(map(add,summ_dic['better'], summ_dic['equal'])), label='worse')
	ax.set_ylabel('Proportions')
	ax.set_title('%s\n%s'%(metric, '/'.join(map(str, summ_counts))), fontsize=4)
	ax.legend()
	plt.tight_layout()
	plt.savefig('%s/summary_plots/%s_%s_test_size_%s.jpg'%(fo_dir, metric, ML, test_size), format='jpg')
	plt.savefig('%s/summary_plots/%s_%s_test_size_%s.eps'%(fo_dir, metric, ML, test_size), format='eps', dpi=300)
	plt.close()


output = pd.DataFrame(data=output, columns=output_col)
output.to_csv('%s/summary_results_%s_test_size_%s.txt'%(fo_dir, ML, test_size), sep='\t', index=False)
