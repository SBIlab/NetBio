## one-to-one, Immune Checkpoint Inihibitor dataset (training dataset) to TCGA (test dataset)
## read README.txt for further details
## This code includes:
##   1. PCA plot for visualization of batch effect removal 
##   2. performance measurements
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut

exec(open('../utilities/parse_patient_data.py').read())
exec(open('../utilities/ML.py').read())
exec(open('../utilities/utilities_for_network_based_ML.py').read())
exec(open('../utilities/parse_TCGA_data.py').read())
exec(open('../utilities/useful_utilities.py').read())



## Initialize
train_datasets = ['Gide', 'Liu', 'Kim', 'IMvigor210']
TCGA_cancer_types = ['SKCM', 'STAD', 'BLCA']
ML_list = ['LogisticRegression']
nGene = 200
qval = 0.01
fo_dir = '../../result/5_ICI_to_TCGA'  #'/home/junghokong/BLCA_cisplatin_immunotherapy/result/4_cross_study_prediction/from_ICI_to_TCGA'
draw_PCA = False


cohort_targets = {'PD1':['Liu', 'Riaz_pre', 'Kim'],
		'PD1_CTLA4':['Gide'],
		'PD-L1':['IMvigor210']}


## Output dataframe
pred_output = defaultdict(list)
pred_output_col = ['train_dataset', 'test_dataset', 'ML', 'test_type', 'nGene', 'qval', 'sample', 'predicted_response', 'predicted_response_proba']


coef_output = defaultdict(list)
coef_output_col = ['train_dataset', 'test_dataset', 'ML', 'nGene', 'qval', 'test_type', 'feature', 'feature_importance', 'abs_feature_importance', 'feature_importance_rank']



# Reactome pathways
reactome = reactome_genes()

# biomarker genes 
bio_df = pd.read_csv('../../data/Marker_summary.txt', sep='\t')
biomarker_dir = '../../result/0_data_collection_and_preprocessing'




## Run cross study predictions
for ML in ML_list:
	for train_idx, train_dataset in enumerate(train_datasets):
		
		### train dataset targets
		for key in list(cohort_targets.keys()):
			if train_dataset in cohort_targets[key]:
				train_targets = key
				break
		target = key
		

		### load datasets
		# load training dataset
		if 'Riaz' in train_dataset:
			train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], drug_treatment=train_dataset.split('_')[1])
		elif 'Prat' in train_dataset:
			train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], Prat_cancer_type=train_dataset.split('_')[1])
		else:
			train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset)
		
		
		
		for cancer_type in TCGA_cancer_types:
			test_dataset = 'TCGA.%s'%cancer_type
		
			# load test dataset
			test_edf, test_epdf = TCGA_gene_expression(cancer_type), TCGA_ssGSEA(cancer_type)
			
			print('\n\n#----------------------------------------')
			print('training: %s, test: %s, %s'%(train_dataset, test_dataset, time.ctime()))


			### data cleanup: match genes and pathways between cohorts
			common_genes, common_pathways = list(set(train_edf['genes'].tolist()) & set(test_edf['genes'].tolist())), list(set(train_epdf['pathway'].tolist()) & set(test_epdf['pathway'].tolist()))
			train_edf = train_edf.loc[train_edf['genes'].isin(common_genes),:].sort_values(by='genes')
			train_epdf = train_epdf.loc[train_epdf['pathway'].isin(common_pathways),:].sort_values(by='pathway')
			test_edf = test_edf.loc[test_edf['genes'].isin(common_genes),:].sort_values(by='genes')
			test_epdf = test_epdf.loc[test_epdf['pathway'].isin(common_pathways),:].sort_values(by='pathway')

			
			### data cleanup: expression standardization
			edf1, edf2, epdf1, epdf2 = train_edf, test_edf, train_epdf, test_epdf
			train_edf = expression_StandardScaler(train_edf)
			train_epdf = expression_StandardScaler(train_epdf)
			test_edf = expression_StandardScaler(test_edf)
			test_epdf = expression_StandardScaler(test_epdf)
		

			
			### network proximal pathways
			proximal_pathways = return_proximal_pathways(train_edf, target, nGene, qval)


			### Train / Test dataset merging
			train_dic = {}
			test_dic = {}
			# 1. NetBio
			train_dic['NetBio'] = train_epdf.loc[train_epdf['pathway'].isin(proximal_pathways),:]
			test_dic['NetBio'] = test_epdf.loc[test_epdf['pathway'].isin(proximal_pathways),:]
			# 2. controls (other biomarkers)
			for test_type, genes in zip(bio_df['Name'].tolist(), bio_df['Gene_list'].tolist()):
				genes = genes.split(':')
				train_dic[test_type] = train_edf.loc[train_edf['genes'].isin(genes),:]
				test_dic[test_type] = test_edf.loc[test_edf['genes'].isin(genes),:]

			
			### Measure Prediction Performances
			print('\tML training & predicting, %s'%time.ctime())
			for test_type in np.append(['NetBio'], bio_df['Name'].tolist()):
				X_train, X_test = train_dic[test_type].T.values[1:], test_dic[test_type].T.values[1:]
				y_train = train_responses
				test_samples = test_dic[test_type].columns[1:]
		

				# make predictions
				model, param_grid = ML_hyperparameters(ML)
				gcv = []
				gcv = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=5).fit(X_train, y_train)
				model = gcv.best_estimator_
				pred_status = gcv.best_estimator_.predict(X_test)
				pred_proba = gcv.best_estimator_.predict_proba(X_test)[:,1]

				# coefficients
				coef_list = gcv.best_estimator_.coef_[0]
				features = train_dic[test_type][train_dic[test_type].columns[0]]
				tmp_coef = pd.DataFrame({'coef':coef_list, 'feature':features, 'abs_coef':list(map(abs, coef_list))})
				feature_ranks = tmp_coef['abs_coef'].rank(ascending=False).tolist()
				for coef, feature, feature_rank in zip(coef_list, features, feature_ranks):
					for key, value in zip(coef_output_col, [train_dataset, test_dataset, ML, nGene, qval, test_type, feature, coef, np.abs(coef), feature_rank]):
						coef_output[key].append(value)
						

				# pred_output
				for sample, pred_response, p_proba in zip(test_samples, pred_status, pred_proba):
					pred_output['train_dataset'].append(train_dataset)
					pred_output['test_dataset'].append(test_dataset)
					pred_output['ML'].append(ML)
					pred_output['test_type'].append(test_type)
					pred_output['nGene'].append(nGene)
					pred_output['qval'].append(qval)
					pred_output['sample'].append(sample)
					pred_output['predicted_response'].append(pred_response)
					pred_output['predicted_response_proba'].append(p_proba)


coef_output = pd.DataFrame(data=coef_output, columns=coef_output_col)
coef_output.to_csv('%s/ICI_to_TCGA_prediction.feature.importances.txt'%fo_dir, sep='\t', index=False)

pred_output = pd.DataFrame(data=pred_output, columns=pred_output_col)
pred_output.to_csv('%s/ICI_to_TCGA_prediction.results.txt'%fo_dir, sep='\t', index=False)
print('Finished, %s'%time.ctime())
