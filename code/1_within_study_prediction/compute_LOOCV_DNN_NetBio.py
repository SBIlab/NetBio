## LOOCV for immunotherapy treated patients using DNN
import pandas as pd
from collections import defaultdict
import numpy as np
import scipy.stats as stat
from statsmodels.stats.multitest import multipletests
import time, os, math, random

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import log_loss, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
loo = LeaveOneOut()
exec(open('../utilities/useful_utilities.py').read())
exec(open('../utilities/parse_patient_data.py').read())
exec(open('../utilities/utilities_for_network_based_ML.py').read())
exec(open('../utilities/ML.py').read())
exec(open('../utilities/DNN.py').read())



## Initialize
ML = 'DNN'; test_type = 'DNN_NetBio'
num_hyperparam_sets = 10
hidden_layers = ['v1', 'v2', 'v3']
i_dropout_rates = [0, 0.1, 0.5]
h_dropout_rates = [0.1, 0.3, 0.5]
epochs = [10, 50, 100, 200]
l2_penalties = [1e-2, 1e-4]
fo_dir = '../../result/1_within_study_prediction'
data_dir = '../../data/ImmunoTherapy'
patNum_cutoff = 1
predict_proba = False # use probability score as a proxy of drug response
dataset_to_test = 'Gide' # ['Gide', 'Liu', 'Kim', 'IMvigor210']

nGene = 200
adj_pval = 0.01
target_dic = {'Gide':'PD1_CTLA4', 'Liu':'PD1', 'Kim':'PD1', 'IMvigor210':'PD-L1'}


## output
output = defaultdict(list)
output_col = []
proximity_df = defaultdict(list)

pred_output = defaultdict(list)
pred_output_col = []



## LOOCV
for fldr in os.listdir(data_dir):
	test_if_true = 0
	if ('et_al' in fldr) or ('IMvigor210' == fldr):
		if (fldr.replace('_et_al','') == dataset_to_test) or (fldr == dataset_to_test):
			test_if_true = 1
		if test_if_true == 0:
			continue
		print('')
		print('%s, %s'%(fldr, time.ctime()))
		study = fldr.split('_')[0]
		testTypes = [] # [ test types ]
		feature_name_dic = defaultdict(list) # { test_type : [ features ] }
		
		## make output directory
		tmp_directories = ['DNN_NetBio', study]
		for tmp_dir in tmp_directories:
			if os.path.isdir('%s/%s'%(fo_dir, tmp_dir)) == False:
				os.mkdir('%s/%s'%(fo_dir, tmp_dir))
			fo_dir = '%s/%s'%(fo_dir, tmp_dir)
		
		
		## load immunotherapy datasets
		edf, epdf = pd.DataFrame(), pd.DataFrame()
		samples, edf, epdf, responses = parse_reactomeExpression_and_immunotherapyResponse(study)


		
		# stop if number of patients are less than cutoff
		if edf.shape[1] < patNum_cutoff:
			continue


				
		# scale (gene expression)
		tmp = StandardScaler().fit_transform(epdf.T.values[1:])
		new_tmp = defaultdict(list)
		new_tmp['pathway'] = epdf['pathway'].tolist()
		for s_idx, sample in enumerate(epdf.columns[1:]):
			new_tmp[sample] = tmp[s_idx]
		epdf = pd.DataFrame(data=new_tmp, columns=epdf.columns)
		
		
		# NetBio pathways		
		NetBio_pathways = return_proximal_pathways(edf, target_dic[study], nGene, adj_pval)
		epdf = epdf.loc[epdf['pathway'].isin(NetBio_pathways),:]
		
		
		# X
		X = epdf.T.values[1:]


		################################################################################> predictions
		### LOOCV
		obs_responses, pred_responses, pred_probabilities = [], [], []
		idx = 0

		for train_idx, test_idx in loo.split(X):
			idx+= 1
			if (idx == 1) or (idx % 5 == 0):
				print('testing %s / %s, %s'%(idx, X.shape[0], time.ctime()))
				if idx != 1:
					print('current accuracy = %s / F1 = %s'%(accuracy_score(obs_responses, pred_responses), f1_score(obs_responses, pred_responses, pos_label=1)))
					print('\n')

			# train test split
			X_train, X_test, y_train, y_test = X[train_idx].astype(np.float), X[test_idx].astype(np.float), responses[train_idx].astype(np.float), responses[test_idx].astype(np.float)
			

			# continue if no feature is available
			if X_train.shape[1] * X_test.shape[1] == 0:
				continue

			# make predictions
			predictions, actual, binary_predictions, loss_, hyperparam_tmp, hyperparam_tmp2 = train_DNN_with_optimization_and_random_select_hyperparam(X_train, y_train, X_test, y_test, hidden_layers, i_dropout_rates, h_dropout_rates, epochs, l2_penalties, num_hyperparam_sets=num_hyperparam_sets)
			if predict_proba == False:
				pred_status = binary_predictions.reshape(-1)[0]
			if predict_proba == True:
				pred_status = predictions.reshape(-1)[0]

			obs_responses.append(y_test[0])
			pred_responses.append(pred_status)
			pred_probabilities.append(predictions.reshape(-1)[0])
			

			# pred_output
			pred_output['study'].append(study)
			pred_output['test_type'].append('DNN')
			pred_output['ML'].append(ML)
			pred_output['sample'].append(np.array(samples)[test_idx][0])
			pred_output['predicted_response'].append(pred_status)
			pred_output['predicted_proba'].append(predictions.reshape(-1)[0])

			
		if len(pred_responses)==0:
			continue

		# predict dataframe
		pdf = defaultdict(list)
		pdf['obs'] = obs_responses
		pdf['pred'] = pred_responses
		pdf = pd.DataFrame(data=pdf, columns=['obs', 'pred'])
		print(pdf)
		
		## output datafraome
		for key, value in zip(['study', 'test_type', 'ML'], [study, test_type, ML]):
			output[key].append(value)
		
		if predict_proba == False:
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
						

output = pd.DataFrame(output)

raise ValueError

proximity_df = pd.DataFrame(proximity_df)
output = pd.DataFrame(data=output, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'accuracy', 'precision', 'recall', 'F1'])
output.to_csv('%s/LOOCV_%s_predictProba_%s.txt'%(fo_dir, ML, predict_proba), sep='\t', index=False)

pred_output = pd.DataFrame(data=pred_output, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'sample', 'predicted_response', 'predicted_proba'])
pred_output.to_csv('%s/LOOCV_%s_predictProba_%s_PredictedResponse.txt'%(fo_dir, ML, predict_proba), sep='\t', index=False)
