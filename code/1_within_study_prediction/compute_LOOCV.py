## LOOCV for immunotherapy treated patients
## compare (1) all genes as features VS (2) immunotherapy biomarker-based network features
import pandas as pd
from collections import defaultdict
import numpy as np
import scipy.stats as stat
from statsmodels.stats.multitest import multipletests
import time, os, math, random

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import *
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
loo = LeaveOneOut()
exec(open('../utilities/useful_utilities.py').read())
exec(open('../utilities/parse_patient_data.py').read())
exec(open('../utilities/ML.py').read())




## Initialize
ML = 'LogisticRegression'  #'LogisticRegression'
fo_dir = '../../result/1_within_study_prediction'
data_dir = '../../data/ImmunoTherapy'
biomarker_dir = '../../result/0_data_collection_and_preprocessing'
patNum_cutoff = 1
num_genes = 200
qval_cutoff = 0.01
predict_proba = False # use probability score as a proxy of drug response
datasets_to_test = ['Kim'] # ['Gide', 'Liu', 'Kim', 'IMvigor210']
target_dic = {'Gide':'PD1_CTLA4', 'Liu':'PD1', 'Kim':'PD1', 'IMvigor210':'PD-L1'}



# Reactome pathways
reactome = reactome_genes()

# biomarker genes
bio_df = pd.read_csv('../../data/Marker_summary.txt', sep='\t')



## output
output = defaultdict(list)
output_col = []
proximity_df = defaultdict(list)

pred_output = defaultdict(list)
pred_output_col = []


## regularization parameter
regularization_param = defaultdict(list)




## LOOCV
for fldr in os.listdir(data_dir):
	test_if_true = 0
	if ('et_al' in fldr) or ('IMvigor210' == fldr):
		if (fldr.replace('_et_al','') in datasets_to_test) or (fldr in datasets_to_test):
			test_if_true = 1
		if test_if_true == 0:
			continue
		print('')
		print('%s, %s'%(fldr, time.ctime()))
		study = fldr.split('_')[0]
		testTypes = [] # [ test types ]
		feature_name_dic = defaultdict(list) # { test_type : [ features ] }
		

		## load immunotherapy datasets
		edf, epdf = pd.DataFrame(), pd.DataFrame()
		_, edf, epdf, responses = parse_reactomeExpression_and_immunotherapyResponse(study)


		
		# stop if number of patients are less than cutoff
		if edf.shape[1] < patNum_cutoff:
			continue

		# scale (gene expression)
		tmp = StandardScaler().fit_transform(edf.T.values[1:])
		new_tmp = defaultdict(list)
		new_tmp['genes'] = edf['genes'].tolist()
		for s_idx, sample in enumerate(edf.columns[1:]):
			new_tmp[sample] = tmp[s_idx]
		edf = pd.DataFrame(data=new_tmp, columns=edf.columns)

		# scale (pathway expression)
		tmp = StandardScaler().fit_transform(epdf.T.values[1:])
		new_tmp = defaultdict(list)
		new_tmp['pathway'] = epdf['pathway'].tolist()
		for s_idx, sample in enumerate(epdf.columns[1:]):
			new_tmp[sample] = tmp[s_idx]
		epdf = pd.DataFrame(data=new_tmp, columns=epdf.columns)

	

		## features, labels
		#exp_dic, responses = defaultdict(list), []
		exp_dic = defaultdict(list)
		sample_dic = defaultdict(list)




		## load biomarker genes
		biomarkers = bio_df['Name'].tolist()
		bp_dic = defaultdict(list) # { biomarker : [ enriched pathways ] } // enriched functions
		for biomarker in biomarkers:
			# biomarker features
			biomarker_genes = bio_df.loc[bio_df['Name']=='%s'%biomarker,:]['Gene_list'].tolist()[0].split(':')
			exp_dic[biomarker] = edf.loc[edf['genes'].isin(biomarker_genes),:].T.values[1:]
			sample_dic[biomarker] = edf.columns[1:]
			feature_name_dic[biomarker] = edf.loc[edf['genes'].isin(biomarker_genes),:]['genes'].tolist()
			testTypes.append(biomarker)
			
			
			if '%s.txt'%biomarker in os.listdir(biomarker_dir):
				# gene expansion by network propagation results
				bdf = pd.read_csv('%s/%s.txt'%(biomarker_dir, biomarker), sep='\t')
				bdf = bdf.dropna(subset=['gene_id'])
				b_genes = []
				for idx, gene in enumerate(bdf.sort_values(by=['propagate_score'], ascending=False)['gene_id'].tolist()):
					if gene in edf['genes'].tolist():
						if not gene in b_genes:
							b_genes.append(gene)
						if len(set(b_genes)) >= num_genes:
							break
				tmp_edf = edf.loc[edf['genes'].isin(b_genes),:]
				
				

				# enriched functions
				tmp_hypergeom = defaultdict(list)
				pvalues, qvalues = [], []; overlap = []; pw_count = []
				for pw in list(reactome.keys()):
					pw_genes = list(set(reactome[pw]) & set(edf['genes'].tolist()))
					M = len(edf['genes'].tolist())
					n = len(pw_genes)
					N = len(set(b_genes))
					k = len(set(pw_genes) & set(b_genes))
					p = stat.hypergeom.sf(k-1, M, n, N)
					tmp_hypergeom['pw'].append(pw)
					tmp_hypergeom['p'].append(p)
					pvalues.append(p)
					overlap.append(k)
					pw_count.append(n)
					proximity_df['biomarker'].append(biomarker)
					proximity_df['study'].append(study)
					proximity_df['pw'].append(pw)
					proximity_df['p'].append(p)
				_, qvalues, _, _ = multipletests(pvalues)
				for q in qvalues:
					proximity_df['q'].append(q)
				tmp_hypergeom['qval'] = qvalues
				tmp_hypergeom['overlap'] = overlap
				tmp_hypergeom['pw_count'] = pw_count
				tmp_hypergeom = pd.DataFrame(tmp_hypergeom)
				bp_dic[biomarker] = tmp_hypergeom.loc[tmp_hypergeom['qval']<=qval_cutoff,:]['pw'].tolist()
				tmp_epdf = epdf.loc[epdf['pathway'].isin(bp_dic[biomarker]),:]
				# NetBio
				if target_dic[study] == biomarker:
					exp_dic['NetBio'] = tmp_epdf.T.values[1:]
					sample_dic['NetBio'] = tmp_epdf.columns[1:]
					feature_name_dic['NetBio'] = tmp_epdf['pathway'].tolist()
					testTypes.append('NetBio')

			


		################################################################################> predictions
		### LOOCV
		selected_feature_dic = {} # { test type : [ test_idx : [ selected features ] ] }

		for test_type in testTypes: # list(exp_dic.keys()):
			print('\n\t%s / test type : %s / ML: %s, %s'%(study, test_type, ML, time.ctime()))
			obs_responses, pred_responses, pred_probabilities = [], [], []
			selected_feature_dic[test_type] = defaultdict(list)

			for train_idx, test_idx in loo.split(exp_dic[test_type]):
				# train test split
				X_train, X_test, y_train, y_test = exp_dic[test_type][train_idx], exp_dic[test_type][test_idx], responses[train_idx], responses[test_idx]
				
				# continue if no feature is available
				if X_train.shape[1] * X_test.shape[1] == 0:
					continue

				# make predictions
				model, param_grid = ML_hyperparameters(ML)
				gcv = []
				gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=5).fit(X_train, y_train)
				
				# regularization param
				regularization_param['study'].append(study)
				regularization_param['test_type'].append(test_type)
				regularization_param['ML'].append(ML)
				regularization_param['nGene'].append(num_genes)
				regularization_param['qval'].append(qval_cutoff)
				regularization_param['test_sample'].append(sample_dic[test_type][test_idx[0]])
				regularization_param['C'].append(gcv.best_params_['C'])

				# predictions
				if predict_proba == False:
					pred_status = gcv.best_estimator_.predict(X_test)[0]
				if predict_proba == True:
					pred_status = gcv.best_estimator_.predict_proba(X_test)[0][1]

				obs_responses.append(y_test[0])
				pred_responses.append(pred_status)
				pred_probabilities.append(gcv.best_estimator_.predict_proba(X_test)[0][1])


				# pred_output
				pred_output['study'].append(study)
				pred_output['test_type'].append(test_type)
				pred_output['ML'].append(ML)
				pred_output['nGene'].append(num_genes)
				pred_output['qval'].append(qval_cutoff)
				pred_output['sample'].append(sample_dic[test_type][test_idx[0]])
				pred_output['predicted_response'].append(pred_status)
				pred_output['pred_proba'].append(gcv.best_estimator_.predict_proba(X_test)[0][1])

			
			if len(pred_responses)==0:
				continue

			# predict dataframe
			pdf = defaultdict(list)
			pdf['obs'] = obs_responses
			pdf['pred'] = pred_responses
			pdf = pd.DataFrame(data=pdf, columns=['obs', 'pred'])
			print(pdf)
			
			## output datafraome
			for key, value in zip(['study', 'test_type', 'ML', 'nGene', 'qval'], [study, test_type, ML, num_genes, qval_cutoff]):
				if key in ['nGene', 'qval']:
					if 'NetBio' in test_type:
						output[key].append(value)
					else:
						output[key].append('na')
				else:
					output[key].append(value)

			# accuracy
			accuracy = accuracy_score(obs_responses, pred_responses)
			output['accuracy'].append(accuracy)
			print('\t%s / accuracy = %s'%(test_type, accuracy))

			# precision
			precision = precision_score(obs_responses, pred_responses, pos_label=1)
			output['precision'].append(precision)
			print('\t%s / precision = %s'%(test_type, precision))

			# recall 
			recall = recall_score(obs_responses, pred_responses, pos_label=1)
			output['recall'].append(recall)
			print('\t%s / recall = %s'%(test_type, recall))

			# F1
			F1 = f1_score(obs_responses, pred_responses, pos_label=1)
			output['F1'].append(F1)
			print('\t%s / F1 = %s'%(test_type, F1))
			
			# TP, TN, FP, FN, sensitivity, specificity
			tn, fp, fn, tp = confusion_matrix(obs_responses, pred_responses).ravel()
			sensitivity = tp/(tp+fn)
			specificity = tn/(tn+fp)
			for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
				output[key].append(value)
				print('\t%s / %s = %s'%(test_type, key, value))
output = pd.DataFrame(output)


proximity_df = pd.DataFrame(proximity_df)
output = pd.DataFrame(data=output, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'accuracy', 'precision', 'recall', 'F1', 'TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'])
output.to_csv('%s/LOOCV_%s_predictProba_%s.txt'%(fo_dir, ML, predict_proba), sep='\t', index=False)

pred_output = pd.DataFrame(data=pred_output, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'sample', 'predicted_response', 'pred_proba'])
pred_output.to_csv('%s/LOOCV_%s_predictProba_%s_PredictedResponse.txt'%(fo_dir, ML, predict_proba), sep='\t', index=False)

regularization_param = pd.DataFrame(data=regularization_param, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'test_sample', 'C'])
regularization_param.to_csv('%s/LOOCV_%s_predictProba_%s_Regularization_param.txt'%(fo_dir, ML, predict_proba), sep='\t', index=False)
