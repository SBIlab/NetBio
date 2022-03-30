## Visualize results
import matplotlib.pyplot as plt
import scipy.stats as stat
import numpy as np
import pandas as pd
from collections import defaultdict
import time, os
from operator import add


## Initialize
ML = 'LogisticRegression'
nGene = 200
adj_pval_cutoff = 0.01
test_datasets = ['Auslander', 'Prat_MELANOMA', 'Riaz_pre']
controls = ['PD1', 'PD-L1', 'CTLA4', 'PD1_PD-L1_CTLA4', 'CD8T1', 'T_exhaust_Pos', 'CAF1', 'TAM_M2_M1_Pos', 'all-TME-Bio']
training_size = 0.8



## Import data & make plots

# draw AUC or AUPRC plot
def draw_AUC_AUPRC(draw_plot_for='AUC', nGene=nGene, adj_pval=adj_pval_cutoff, controls=['PD1'], ML=ML):
	# output directory
	plt_dir = '../../result/2_cross_study_prediction'
	tmp_directories = ['%s_plots'%draw_plot_for, 'nGene_%s_adj_pval_%s'%(nGene, adj_pval)]
	for tdir in tmp_directories:
		if os.path.isdir('%s/%s'%(plt_dir, tdir)) == False:
			os.mkdir('%s/%s'%(plt_dir, tdir))
		plt_dir = '%s/%s'%(plt_dir, tdir)
	# import data
	df = pd.read_csv('../../result/2_cross_study_prediction/across_study_performance.txt', sep='\t')
	for train in df['train_dataset'].value_counts().index:
		for test in df['test_dataset'].value_counts().index:
			plt.figure(figsize=(8,8))
			plt.xlim(-0.05, 1.05)
			plt.ylim(-0.05, 1.05)
			for test_type in np.append(['NetBio'], controls):
				tmp = df.loc[df['train_dataset']==train,:].loc[df['test_dataset']==test,:].loc[df['ML']==ML,:].loc[df['test_type']==test_type,:].loc[df['nGene']==nGene,:].loc[df['qval']==adj_pval,:]
				print(pd.DataFrame(data=tmp, columns=['train_dataset', 'test_dataset', 'test_type', 'AUC_proba', 'AUPRC']))
				print('#------------\n\n')
				if draw_plot_for == 'AUC':
					fpr = list(map(float, tmp['fpr_proba'].tolist()[0].split(',')))
					tpr = list(map(float, tmp['tpr_proba'].tolist()[0].split(',')))
					plt.plot([0,1], [0,1], 'k--')
					plt.plot(fpr, tpr, label='%s (AUC=%.3f)'%(test_type, tmp['AUC_proba'].tolist()[0]))
					xlabel = 'False-positive rate'
					ylabel = 'True-positive rate'
				if draw_plot_for == 'AUPRC':
					precision = list(map(float, tmp['precisions'].tolist()[0].split(',')))
					recall = list(map(float, tmp['recalls'].tolist()[0].split(',')))
					plt.plot(recall, precision, label='%s (AUPRC=%.3f)'%(test_type, tmp['AUPRC'].tolist()[0]))
					xlabel = 'Recall'
					ylabel = 'Precision'
			plt.legend(loc='lower right')
			plt.title('train: %s, test: %s'%(train, test))
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.tight_layout()
			plt.savefig('%s/%s_%s.to.%s.jpg'%(plt_dir, ML, train, test), format='jpg')
			plt.savefig('%s/%s_%s.to.%s.eps'%(plt_dir, ML, train, test), format='eps', dpi=300)
			plt.close()



# draw barplots (AUC or AUPRC)
def draw_AUC_AUPRC_barplots(draw_plot_for='AUC', nGene=nGene, adj_pval=adj_pval_cutoff, controls=['PD1'], ML=ML):
	# output directory
	plt_dir = '../../result/2_cross_study_prediction'
	tmp_directories = ['%s_plots'%draw_plot_for, 'nGene_%s_adj_pval_%s'%(nGene, adj_pval)]
	for tdir in tmp_directories:
		if os.path.isdir('%s/%s'%(plt_dir, tdir)) == False:
			os.mkdir('%s/%s'%(plt_dir, tdir))
		plt_dir = '%s/%s'%(plt_dir, tdir)
	# import data
	df = pd.read_csv('../../result/2_cross_study_prediction/across_study_performance.txt', sep='\t')
	for train in df['train_dataset'].value_counts().index:
		for test in df['test_dataset'].value_counts().index:
			plt.figure(figsize=(8,8))
			for test_type in np.append(['NetBio'], controls):
				tmp = df.loc[df['train_dataset']==train,:].loc[df['test_dataset']==test,:].loc[df['ML']==ML,:].loc[df['test_type']==test_type,:].loc[df['nGene']==nGene,:].loc[df['qval']==adj_pval,:]
				print(pd.DataFrame(data=tmp, columns=['train_dataset', 'test_dataset', 'test_type', 'AUC_proba', 'AUPRC']))
				print('#------------\n\n')
				if draw_plot_for == 'AUC':
					fpr = list(map(float, tmp['fpr_proba'].tolist()[0].split(',')))
					tpr = list(map(float, tmp['tpr_proba'].tolist()[0].split(',')))
					plt.plot([0,1], [0,1], 'k--')
					plt.plot(fpr, tpr, label='%s (AUC=%.3f)'%(test_type, tmp['AUC_proba'].tolist()[0]))
					xlabel = 'False-positive rate'
					ylabel = 'True-positive rate'
				if draw_plot_for == 'AUPRC':
					precision = list(map(float, tmp['precisions'].tolist()[0].split(',')))
					recall = list(map(float, tmp['recalls'].tolist()[0].split(',')))
					plt.plot(recall, precision, label='%s (AUPRC=%.3f)'%(test_type, tmp['AUPRC'].tolist()[0]))
					xlabel = 'Recall'
					ylabel = 'Precision'
			plt.legend(loc='lower right')
			plt.title('train: %s, test: %s'%(train, test))
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.tight_layout()
			plt.savefig('%s/%s_%s.to.%s.jpg'%(plt_dir, ML, train, test), format='jpg')
			plt.savefig('%s/%s_%s.to.%s.eps'%(plt_dir, ML, train, test), format='eps', dpi=300)
			plt.close()



# draw barplots and boxplots for reduced training set predictions
def draw_reduced_training(draw_plot_for='AUC', pval_cutoff=0.05, nGene=nGene, adj_pval=adj_pval_cutoff, controls=controls, ML=ML):
	output = defaultdict(list)
	output_col = ['train_dataset', 'test_dataset', 'metric',  'control', 'NetPw_mean', 'control_mean', 'ttest_pvalue', 'prediction_result']
	fo_dir = '../../result/2_cross_study_prediction/reduced_training_dataset_results'
	df = pd.read_csv('../../result/2_cross_study_prediction/across_study_prediction_training_size_%s.txt'%training_size, sep='\t')


	for train in df['train_dataset'].value_counts().index:
		train_test = []
		summ_dic = defaultdict(list)
		summ_counts = []
		for test in test_datasets:
			train_test.append('%s.to.%s'%(train, test))
			# dataframe
			temp = df.loc[df['train_dataset']==train,:].loc[df['test_dataset']==test,:].loc[df['ML']==ML,:].loc[df['nGene']==nGene,:].loc[df['qval']==adj_pval,:]
			# NetBio
			if draw_plot_for == 'AUC':
				NetBio = temp.loc[temp['test_type']=='NetBio',:]['AUC_proba'].tolist()
			if draw_plot_for == 'AUPRC':
				NetBio = temp.loc[temp['test_type']=='NetBio',:]['AUPRC'].tolist()
			# controls
			cScores = []
			for control in controls:
				if draw_plot_for == 'AUC':
					scores = temp.loc[temp['test_type']==control,:]['AUC_proba'].tolist()
				if draw_plot_for == 'AUPRC':
					scores = temp.loc[temp['test_type']==control,:]['AUPRC'].tolist()
				if len(scores) > 0:
					cScores.append(scores)
				else:
					cScores.append(np.zeros(len(NetBio)))

			# 1. summarized prediction results (barplot)
			summ = {'better':0, 'equal':0, 'worse':0}
			for control, scores in zip(controls, cScores):
				# test statistical significance
				_, pval = stat.ttest_ind(NetBio, scores)
				for key, value in zip(output_col[:-1], [train, test, draw_plot_for, control, np.mean(NetBio), np.mean(scores), pval]):
					output[key].append(value)
				if pval <= pval_cutoff:
					if np.mean(NetBio) > np.mean(scores):
						summ['better']+=1
						output['prediction_result'].append('better')
					else:
						summ['worse']+=1
						output['prediction_result'].append('worse')
				else:
					summ['equal']+= 1
					output['prediction_result'].append('equal')
			for test_result in ['better', 'equal', 'worse']:
				percent_score = 100*summ[test_result]/len(cScores)
				summ_dic[test_result].append(percent_score)
				summ_counts.append('%s_%s_%s_%s'%(train, test, test_result, summ[test_result]))

			# 2. boxplot
			all_scores = np.append([NetBio], cScores, axis=0)
			labels = np.append(['NetBio'], controls)
			plt.close()
			plt.boxplot(list(all_scores))
			plt.xticks(np.arange(1,len(labels)+1), labels, rotation=90)
			plt.ylabel(draw_plot_for)
			plt.xlabel('Features')
			plt.tight_layout()
			plt.savefig('%s/boxplots/%s_to_%s_%s_metric_%s_trainSize_%s.jpg'%(fo_dir, train, test, ML, draw_plot_for, training_size), format='jpg')
			plt.savefig('%s/boxplots/%s_to_%s_%s_metric_%s_trainSize_%s.eps'%(fo_dir, train, test, ML, draw_plot_for, training_size), format='eps', dpi=300)
			plt.close()
		# 3. summary bar plot
		plt.close()
		aig, ax = plt.subplots()
		ax.bar(train_test, summ_dic['better'], label='better')
		ax.bar(train_test, summ_dic['equal'], bottom=summ_dic['better'], label='equal')
		ax.bar(train_test, summ_dic['worse'], bottom=list(map(add, summ_dic['better'], summ_dic['equal'])), label='worse')
		ax.set_ylabel('Proportions')
		ax.set_title('%s\n%s'%(draw_plot_for, '/'.join(map(str, summ_counts))), fontsize=4)
		ax.legend()
		plt.tight_layout()
		plt.savefig('%s/summary_plots/trainingData_%s_%s_metric_%s_trainSize_%s.jpg'%(fo_dir, train, ML, draw_plot_for, training_size), format='jpg')
		plt.savefig('%s/summary_plots/trainingData_%s_%s_metric_%s_trainSize_%s.eps'%(fo_dir, train, ML, draw_plot_for, training_size), format='eps', dpi=300)
		plt.close()
	output = pd.DataFrame(data=output, columns=output_col)
	output.to_csv('%s/summary_results_%s_metric_%s_trainSize_%s.txt'%(fo_dir, ML, draw_plot_for, training_size), sep='\t', index=False)
	return summ_dic
