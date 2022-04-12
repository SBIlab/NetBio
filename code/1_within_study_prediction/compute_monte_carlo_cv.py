## monte carlo cross validation for immunotherapy treated patients
## compare (1) all genes as features VS (2) immunotherapy biomarker-based network features
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



## Initialize
ML = 'LogisticRegression'  
fo_dir = '../../result/1_within_study_prediction'
data_dir = '../../data/ImmunoTherapy'
biomarker_dir = '../../result/0_data_collection_and_preprocessing'
patNum_cutoff = 1
test_size = 0.2
iternum = 100
num_genes = 200
qval_cutoff = 0.01
datasets_to_test = ['Kim'] # ['Gide', 'Liu', 'Kim', 'IMvigor210'] 
target_dic = {'Gide':'PD1_CTLA4', 'Liu':'PD1', 'Kim':'PD1', 'IMvigor210':'PD-L1'}



# Reactome pathways
reactome = reactome_genes()

# biomarker genes
bio_df = pd.read_csv('../../data/Marker_summary.txt', sep='\t')



## output
output = defaultdict(list)
output_col = []






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
		exp_dic = defaultdict(list)
		sample_dic = defaultdict(list)


		## load biomarker genes
		biomarkers = bio_df['Name'].tolist()
		bp_dic = defaultdict(list) # { biomarker : [ enriched pathways ] } // enriched functions
		for biomarker in biomarkers:
			# biomarkers
			biomarker_genes = bio_df.loc[bio_df['Name']=='%s'%biomarker,:]['Gene_list'].tolist()[0].split(':')
			exp_dic[biomarker] = edf.loc[edf['genes'].isin(biomarker_genes),:].T.values[1:]
			sample_dic[biomarker] = edf.columns[1:]
			feature_name_dic[biomarker] = edf.loc[edf['genes'].isin(biomarker_genes),:]['genes'].tolist()
			testTypes.append(biomarker)
			
			# NetBio pathways
			if '%s.txt'%biomarker in os.listdir(biomarker_dir):
				if biomarker == target_dic[study]:
					# proximal pathways
					proximal_pathways = return_proximal_pathways(edf, biomarker, num_genes, qval_cutoff)
					bp_dic[biomarker] = proximal_pathways
					tmp_epdf = epdf.loc[epdf['pathway'].isin(bp_dic[biomarker]),:]
					# biomarker network module features
					exp_dic['NetBio'] = tmp_epdf.T.values[1:]
					sample_dic['NetBio'] = tmp_epdf.columns[1:]
					feature_name_dic['NetBio'] = tmp_epdf['pathway'].tolist()
					testTypes.append('NetBio')
	



		################################################################################> predictions
		### monte carlo CV
		selected_feature_dic = {} # { test type : [ test_idx : [ selected features ] ] }

		for test_type in testTypes: # list(exp_dic.keys()):
			print('\n\t%s / test type : %s / ML: %s, %s'%(study, test_type, ML, time.ctime()))
			obs_responses, pred_responses = [], []
			selected_feature_dic[test_type] = defaultdict(list)

			#for train_idx, test_idx in loo.split(exp_dic[test_type]):
			for i in range(iternum):
				i += 1
				# train test split
				X_train, X_test, y_train, y_test = train_test_split(exp_dic[test_type], responses, test_size=test_size, random_state=i, stratify=responses)
				
				# continue if no feature is available
				if X_train.shape[1] * X_test.shape[1] == 0:
					continue

								
				
				# make predictions
				model, param_grid = ML_hyperparameters(ML) #model_selector(ML)
				gcv = []
				gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=5).fit(X_train, y_train)
				pred_status = gcv.best_estimator_.predict(X_test)
				obs_responses, pred_responses = y_test, pred_status

			
				if len(pred_responses)==0:
					continue
				
				# output dataframe
				for key, value in zip(['study', 'iternum', 'test_size', 'test_type', 'ML', 'nGene', 'qval'], [study, i, test_size, test_type, ML, num_genes, qval_cutoff]):
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

