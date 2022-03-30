## LOOCV for immunotherapy treated patients, using TMB and network-based transcriptome features
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
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LogisticRegression, LinearRegression, RidgeClassifier
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
loo = LeaveOneOut()
exec(open('../utilities/ML.py').read())
exec(open('../utilities/parse_patient_data.py').read())
exec(open('../utilities/useful_utilities.py').read())




## Initialize
ML = 'LogisticRegression' # 'SVC' 'RandomForest', 'LogisticRegression', 'RidgeClassifier'
fo_dir = '../../result/3_NetBio_plus_TMB'
data_dir = '../../data/ImmunoTherapy'
biomarker_dir = '../../result/0_data_collection_and_preprocessing'
patNum_cutoff = 1
num_genes = 200
qval_cutoff = 0.01
datasets_to_test = ['Liu', 'IMvigor210']
cFeature = 'TMB'



# Reactome pathways
reactome = reactome_genes()

# biomarker genes
#bio_df = pd.read_csv('/home/user/shared/ImmunoTherapy/Marker_summary.txt', sep='\t')
bio_df = pd.read_csv('../../data/Marker_summary.txt', sep='\t')

# cohort targets
cohort_targets = {'PD1':['Liu'], 'PD-L1':['IMvigor210']}


## output
proximity_df = defaultdict(list)

pred_output = defaultdict(list)
pred_output_col = []




## LOOCV
for fldr in os.listdir(data_dir):
	test_if_true = 0
	if ('et_al' in fldr) or ('IMvigor210' == fldr):

		### 
		if (fldr.replace('_et_al','') in datasets_to_test) or (fldr in datasets_to_test):
			test_if_true = 1
		if test_if_true == 0:
			continue
		print('')
		print('%s, %s'%(fldr, time.ctime()))
		study = fldr.split('_')[0]
		testTypes = [] # [ test types ]
		feature_name_dic = defaultdict(list) # { test_type : [ features ] }
		
		### train dataset targets
		for key in list(cohort_targets.keys()):
			if study in cohort_targets[key]:
				train_targets = key
				break
		target = key

		## load immunotherapy datasets
		edf, epdf = pd.DataFrame(), pd.DataFrame()
		samples, edf, epdf, responses = parse_reactomeExpression_and_immunotherapyResponse(study)

		## load clinical features
		pdf = parse_clinical_features(study, clinical_feature=cFeature)

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

		
		## reorder data
		samples2, edf2, epdf2, responses2 = [], defaultdict(list), defaultdict(list), []
		for sample, response in zip(samples, responses):
			if sample in pdf['sample'].tolist():
				samples2.append(sample)
				responses2.append(response)
		edf2 = pd.DataFrame(data=edf, columns=np.append(['genes'], samples2))
		epdf2 = pd.DataFrame(data=epdf, columns=np.append(['pathway'], samples2))
		# reorder
		samples = samples2
		edf = edf2
		epdf = epdf2
		responses = np.array(responses2)


		## features, labels
		#exp_dic, responses = defaultdict(list), []
		exp_dic = defaultdict(list)
		sample_dic = defaultdict(list)

		# clinical features
		exp_dic[cFeature] = []
		for sample in samples:
			exp_dic[cFeature].append([pdf.loc[pdf['sample']==sample,:]['status'].tolist()[0]])
			sample_dic[cFeature].append(sample)
		exp_dic[cFeature] = StandardScaler().fit_transform(np.array(exp_dic[cFeature]))
		testTypes.append(cFeature)


		## load network-based features
		# gene expansion by network propagation results
		bdf = pd.read_csv('%s/%s.txt'%(biomarker_dir, target), sep='\t')
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
		pvalues, qvalues = [], []
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
			proximity_df['biomarker'].append(target)
			proximity_df['study'].append(study)
			proximity_df['pw'].append(pw)
			proximity_df['p'].append(p)
		_, qvalues, _, _ = multipletests(pvalues)
		for q in qvalues:
			proximity_df['q'].append(q)
		tmp_hypergeom['qval'] = qvalues
		tmp_hypergeom = pd.DataFrame(tmp_hypergeom)
		proximal_pathways = tmp_hypergeom.loc[tmp_hypergeom['qval']<=qval_cutoff,:]['pw'].tolist()
		tmp_epdf = epdf.loc[epdf['pathway'].isin(proximal_pathways),:]
		# biomarker network module features
		exp_dic['%s_NetBio'%target] = tmp_epdf.T.values[1:]
		sample_dic['%s_NetBio'%target] = tmp_epdf.columns[1:]
		feature_name_dic['%s_NetBio'%target] = tmp_epdf['pathway'].tolist()
		testTypes.append('%s_NetBio'%target)


		
		### TMB + NetBio
		exp_dic['%s.plus.%s_NetBio'%(cFeature, target)] = np.append(exp_dic[cFeature], exp_dic['%s_NetBio'%target], axis=1)
		testTypes.append('%s.plus.%s_NetBio'%(cFeature, target))
		sample_dic['%s.plus.%s_NetBio'%(cFeature, target)] = samples
		

		################################################################################> predictions
		### LOOCV
		selected_feature_dic = {} # { test type : [ test_idx : [ selected features ] ] }
		pred_models = {} # { test_type : { sample : model } }

		for test_type in testTypes:
			print('\n\t%s / test type : %s / ML: %s, %s'%(study, test_type, ML, time.ctime()))
			obs_responses, pred_responses = [], []
			selected_feature_dic[test_type] = defaultdict(list)
			pred_models[test_type] = {}

			for train_idx, test_idx in loo.split(exp_dic[test_type]):
				X_train, X_test, y_train, y_test = exp_dic[test_type][train_idx], exp_dic[test_type][test_idx], responses[train_idx], responses[test_idx]
				test_sample = samples[test_idx[0]]

				
				# continue if no feature is available
				if X_train.shape[1] * X_test.shape[1] == 0:
					continue

				
				# make predictions
				model, param_grid = ML_hyperparameters(ML)
				gcv = []
				gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=5).fit(X_train, y_train)
				pred_status = gcv.best_estimator_.predict(X_test)[0]
				pred_proba = gcv.best_estimator_.predict_proba(X_test)[0][1]
				pred_models[test_type][test_sample] = gcv.best_estimator_

				obs_responses.append(y_test[0])
				pred_responses.append(pred_status)

				# pred_output
				pred_output['study'].append(study)
				pred_output['test_type'].append(test_type)
				pred_output['ML'].append(ML)
				pred_output['nGene'].append(num_genes)
				pred_output['qval'].append(qval_cutoff)
				pred_output['sample'].append(sample_dic[test_type][test_idx[0]])
				pred_output['pred_proba'].append(pred_proba)



pred_output = pd.DataFrame(data=pred_output, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'sample', 'pred_proba'])
pred_output.to_csv('%s/LOOCV_%s_PredictedResponse.txt'%(fo_dir, ML), sep='\t', index=False)
