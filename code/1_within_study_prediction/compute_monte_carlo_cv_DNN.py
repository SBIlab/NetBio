## monte carlo cross validation for immunotherapy treated patients
import pandas as pd
from collections import defaultdict
import numpy as np
import scipy.stats as stat
from statsmodels.stats.multitest import multipletests
import time, os, math, random

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import log_loss, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LogisticRegression, LinearRegression, RidgeClassifier
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
loo = LeaveOneOut()
exec(open('../utilities/useful_utilities.py').read())
exec(open('../utilities/parse_patient_data.py').read())
exec(open('../utilities/ML.py').read())
exec(open('../utilities/utilities_for_network_based_ML.py').read())
exec(open('../utilities/DNN.py').read())




## Initialize
ML = 'DNN'; test_type = 'DNN'
num_hyperparam_sets = 10
hidden_layers = ['v1', 'v2', 'v3']
i_dropout_rates = [0, 0.1, 0.5]
h_dropout_rates = [0.1, 0.3, 0.5]
epochs = [10, 50, 100, 200]
l2_penalties = [1e-2, 1e-4]
data_dir = '../../data/ImmunoTherapy'
patNum_cutoff = 1
test_size = 0.2
iternum = 100
dataset_to_test = 'Kim' # ['Gide', 'Liu', 'Kim', 'IMvigor210'] 



## output
output = defaultdict(list)
output_col = []



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
		
		## make output directory
		fo_dir = '../../result/1_within_study_prediction'
		tmp_directories = ['DNN', study]
		for tmp_dir in tmp_directories:
			if os.path.isdir('%s/%s'%(fo_dir, tmp_dir)) == False:
				os.mkdir('%s/%s'%(fo_dir, tmp_dir))
			fo_dir = '%s/%s'%(fo_dir, tmp_dir)



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
	

		## X
		X = edf.T.values[1:]

	



		################################################################################> predictions
		### monte carlo CV
		print('\n\t%s / test type : %s / ML: %s, %s'%(study, test_type, ML, time.ctime()))
		obs_responses, pred_responses = [], []

		for i in range(iternum):
			i += 1
			if (i == 1) or (i % 10 == 0):
				print('\tmonte carlo %s / %s, %s'%(i, iternum, time.ctime()))

			# train test split
			X_train, X_test, y_train, y_test = train_test_split(X, responses, test_size=test_size, random_state=i, stratify=responses)
			X_train, X_test, y_train, y_test = X_train.astype(np.float), X_test.astype(np.float), y_train.astype(np.float), y_test.astype(np.float)
			
			# continue if no feature is available
			if X_train.shape[1] * X_test.shape[1] == 0:
				continue

							
			
			# make predictions
			predictions, actual, binary_predictions, loss_, hyperparam_tmp, hyperparam_tmp2 = train_DNN_with_optimization_and_random_select_hyperparam(X_train, y_train, X_test, y_test, hidden_layers, i_dropout_rates, h_dropout_rates, epochs, l2_penalties, num_hyperparam_sets=num_hyperparam_sets)
			obs_responses, pred_responses = y_test, binary_predictions


		
			if len(pred_responses)==0:
				continue
			
			# output dataframe
			for key, value in zip(['study', 'iternum', 'test_size', 'test_type'], [study, i, test_size, ML]):
				output[key].append(value)

			# accuracy
			accuracy = accuracy_score(obs_responses, pred_responses)
			output['accuracy'].append(accuracy)

			# precision
			precision = precision_score(obs_responses, pred_responses, pos_label=1)
			output['precision'].append(precision)

			# recall 
			recall = recall_score(obs_responses, pred_responses, pos_label=1)
			output['recall'].append(recall)
			
			# F1
			F1 = f1_score(obs_responses, pred_responses, pos_label=1)
			output['F1'].append(F1)

output = pd.DataFrame(data=output, columns=['study', 'iternum', 'test_size', 'test_type', 'ML', 'nGene', 'qval', 'accuracy', 'precision', 'recall', 'F1'])
output.to_csv('%s/monteCarlo_testsize_%s_%s.txt'%(fo_dir, test_size, ML), sep='\t', index=False)

