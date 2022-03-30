## visualize LOOCV results
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
from collections import defaultdict
import pandas as pd
import os, time
exec(open('../utilities/parse_patient_data.py').read())


## output directory
fo_dir = '../../result/1_within_study_prediction/LOOCV_plots'



## metrics
metrics = ['accuracy', 'F1']


## load LOOCV results
df = pd.read_csv('../../result/1_within_study_prediction/LOOCV_LogisticRegression_predictProba_False.txt', sep='\t')

## drug target info
targetDic = {'Gide':'PD1_CTLA4', 'Liu':'PD1', 'Kim':'PD1', 'IMvigor210':'PD-L1'}

## comparisons
comparisons = ['PD1', 'PD-L1', 'CTLA4', 'PD1_PD-L1_CTLA4', 'CD8T1', 'T_exhaust_Pos', 'CAF1', 'TAM_M2_M1_Pos', 'all-TME-Bio']


## visualize results
def barplots():
	for metric in metrics:
		for study in ['Gide', 'Liu', 'Kim', 'IMvigor210']:
			scores = []
			xlabels = []
			target = targetDic[study]
			all_comparisons = np.append(['%s_NetBio'%target], comparisons)
			for test_type in all_comparisons:
				xlabels.append(test_type)
				score = df.loc[df['study']==study,:].loc[df['test_type']==test_type,:][metric].tolist()[0]
				scores.append(score)
			# draw figure
			plt.figure(figsize=(8,8))
			plt.bar(np.arange(len(scores)), scores)
			plt.xticks(np.arange(len(scores)), xlabels, rotation=90)
			plt.title(study)
			plt.ylabel(metric)
			plt.tight_layout()
			plt.savefig('%s/%s_%s.jpg'%(fo_dir, study, metric), format='jpg')
			plt.savefig('%s/%s_%s.eps'%(fo_dir, study, metric), format='eps', dpi=300)
			plt.close()	






def LOOCV_stacked_barplots(dataset, biomarker):
	df = pd.read_csv('../../result/1_within_study_prediction/LOOCV_LogisticRegression_predictProba_False_PredictedResponse.txt', sep='\t')
	samples, edf, epdf, responses = parse_reactomeExpression_and_immunotherapyResponse(dataset)
	pdf = pd.DataFrame({'sample':samples, 'response':responses})
	merged = pd.merge(df, pdf, on='sample', how='inner')
	merged = merged.loc[merged['test_type']==biomarker,:]
	if merged.shape[0]==0:
		print('provide correct dataset or biomarker')
		raise ValueError
	# TP, TN, FP, FN
	TP = merged.loc[merged['response']==1,:]['predicted_response'].tolist().count(1)
	TN = merged.loc[merged['response']==0,:]['predicted_response'].tolist().count(0)
	FP = merged.loc[merged['response']==0,:]['predicted_response'].tolist().count(1)
	FN = merged.loc[merged['response']==1,:]['predicted_response'].tolist().count(0)
	_, pval = stat.fisher_exact([[TP, FP], [FN, TN]])
	responder, nonresponder = [], [] # [ responder, nonresponder ]
	responder = [ (TP/(TP+FP))*100, (FN/(TN+FN))*100 ]# (FP/(TP+FP))*100 ]
	nonresponder = [ (FP/(TP+FP))*100, (TN/(TN+FN))*100 ]
	fig, ax = plt.subplots()
	width = 0.35
	labels = ['Pred R\n(n=%s)'%merged['predicted_response'].tolist().count(1), 'Pred NR\n(n=%s)'%merged['predicted_response'].tolist().count(0)]
	ax.bar(labels, responder, width, label='Responder (n=%s)'%merged['response'].tolist().count(1))
	ax.bar(labels, nonresponder, width, bottom=responder, label='NonResonder (n=%s)'%merged['response'].tolist().count(0))
	ax.legend(loc='upper right')
	ax.set_ylabel('Percent of class (%)')
	ax.set_title('%s/ %s / pvalue = %s'%(dataset, biomarker, pval))
	plt.tight_layout()
	#plt.show()
	plt.savefig('../../result/1_within_study_prediction/LOOCV_plots/%s_stacked_barplot_%s.jpg'%(dataset, biomarker), format='jpg')
	plt.savefig('../../result/1_within_study_prediction/LOOCV_plots/%s_stacked_barplot_%s.eps'%(dataset, biomarker), format='eps', dpi=300)
	plt.close()

