## one-to-one, one-step cross study predictions
## read README.txt for further details
## This code uses reduced number of training set to make predictions 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut
import random

exec(open('../utilities/parse_patient_data.py').read())
exec(open('../utilities/ML.py').read())
exec(open('../utilities/useful_utilities.py').read())



## Initialize
studies = [ 'Prat_MELANOMA', 'Riaz_pre', 'Auslander', 'Huang_pre']
cohort_targets = {'PD1_CTLA4':['Gide'], 'PD1':['Liu']}
ML_list = ['LogisticRegression']
training_size = 0.8
iternum = 100
nGene = 200
qval = 0.01
fo_dir = '../../result/2_cross_study_prediction'


train_datasets = []
for key in list(cohort_targets.keys()):
	for value in cohort_targets[key]:
		train_datasets.append(value)


## Output dataframe
output = defaultdict(list)
output_col = ['train_dataset', 'test_dataset', 'iternum', 'training_size', 'training_sample_num', 'nGene', 'qval', 'ML', 'test_type', 'AUC_proba', 'fpr_proba', 'tpr_proba', 'AUPRC', 'expected_AUPRC', 'precisions', 'recalls']




# Reactome pathways
reactome = reactome_genes()

# biomarker genes 
bio_df = pd.read_csv('../../data/Marker_summary.txt', sep='\t')
biomarker_dir = '../../result/0_data_collection_and_preprocessing'




## Run cross study predictions

for ML in ML_list:
	for train_idx, train_dataset in enumerate(train_datasets):
		for test_idx, test_dataset in enumerate(studies):
			### train dataset targets
			for key in list(cohort_targets.keys()):
				if train_dataset in cohort_targets[key]:
					train_targets = key
					break
			target = key
			

			### load datasets
			if ('Riaz' in train_dataset) or ('Huang' in train_dataset):
				train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], drug_treatment=train_dataset.split('_')[1])
			elif 'Prat' in train_dataset:
				train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], Prat_cancer_type=train_dataset.split('_')[1])
			else:
				train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset)
			if ('Riaz' in test_dataset) or ('Huang' in test_dataset):
				test_samples, test_edf, test_epdf, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset.split('_')[0], drug_treatment=test_dataset.split('_')[1])
			elif 'Prat' in test_dataset:
				test_samples, test_edf, test_epdf, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset.split('_')[0], Prat_cancer_type=test_dataset.split('_')[1])
			else:
				test_samples, test_edf, test_epdf, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset)


			### requirements
			if train_dataset == test_dataset:
				continue
			if ('Riaz' in train_dataset) and ('Riaz' in test_dataset):
				continue
			if (len(train_samples) < 30) or (train_responses.tolist().count(1) < 10) or (train_responses.tolist().count(0) < 10):
				continue
			
			print('\n\n#----------------------------------------')
			print('training: %s, test: %s, %s'%(train_dataset, test_dataset, time.ctime()))


			### data cleanup: match genes and pathways between cohorts
			#print(train_edf.shape, train_epdf.shape, test_edf.shape, test_epdf.shape)
			common_genes, common_pathways = list(set(train_edf['genes'].tolist()) & set(test_edf['genes'].tolist())), list(set(train_epdf['pathway'].tolist()) & set(test_epdf['pathway'].tolist()))
			train_edf = train_edf.loc[train_edf['genes'].isin(common_genes),:]
			train_epdf = train_epdf.loc[train_epdf['pathway'].isin(common_pathways),:]
			test_edf = test_edf.loc[test_edf['genes'].isin(common_genes),:]
			test_epdf = test_epdf.loc[test_epdf['pathway'].isin(common_pathways),:]
			#print(train_edf.shape, train_epdf.shape, test_edf.shape, test_epdf.shape)


			### data cleanup: expression standardization
			edf1, edf2, epdf1, epdf2 = train_edf, test_edf, train_epdf, test_epdf
			train_edf = expression_StandardScaler(train_edf).sort_values(by='genes')
			train_epdf = expression_StandardScaler(train_epdf).sort_values(by='pathway')
			test_edf = expression_StandardScaler(test_edf).sort_values(by='genes')
			test_epdf = expression_StandardScaler(test_epdf).sort_values(by='pathway')
		


			
			### network proximal pathways
			# gene expansion by network propagation results
			bdf = pd.read_csv('%s/%s.txt'%(biomarker_dir, key), sep='\t')
			bdf = bdf.dropna(subset=['gene_id'])
			b_genes = []
			for idx, gene in enumerate(bdf.sort_values(by=['propagate_score'], ascending=False)['gene_id'].tolist()):
				if gene in train_edf['genes'].tolist():
					if not gene in b_genes:
						b_genes.append(gene)
					if len(set(b_genes)) >= nGene:
						break
			# LCC function enrichment
			tmp_hypergeom = defaultdict(list)
			pvalues, qvalues = [], []
			for pw in list(reactome.keys()):
				pw_genes = list(set(reactome[pw]) & set(train_edf['genes'].tolist()))
				M = len(train_edf['genes'].tolist())
				n = len(pw_genes)
				N = len(set(b_genes))
				k = len(set(pw_genes) & set(b_genes))
				p = stat.hypergeom.sf(k-1, M, n, N)
				tmp_hypergeom['pw'].append(pw)
				tmp_hypergeom['p'].append(p)
				pvalues.append(p)
			_, qvalues, _, _ = multipletests(pvalues)
			tmp_hypergeom['q'] = qvalues
			tmp_hypergeom = pd.DataFrame(tmp_hypergeom).sort_values(by=['q'])
			proximal_pathways = tmp_hypergeom.loc[tmp_hypergeom['q']<=qval,:]['pw'].tolist() ## proximal_pathways


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
				tmp_X_train, X_test = train_dic[test_type].T.values[1:], test_dic[test_type].T.values[1:]
				tmp_y_train, y_test = train_responses, test_responses

				AUCs = []; AUPRCs = []
				for i in range(iternum):
					i += 1
					# randomly sample from training data
					random.seed(i)
					random_index = random.sample(list(np.arange(len(tmp_y_train))), round(len(tmp_y_train) * training_size))
					X_train = tmp_X_train[random_index]
					y_train = tmp_y_train[random_index]

					# 
					if X_train.shape[1] * X_train.shape[0] * len(y_train) * len(y_test) == 0:
						continue

					# make predictions
					model, param_grid = ML_hyperparameters(ML)
					gcv = []
					gcv = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=5).fit(X_train, y_train)
					model = gcv.best_estimator_
					pred_status = gcv.best_estimator_.predict(X_test)
					pred_proba = gcv.best_estimator_.predict_proba(X_test)[:,1]


					#### measure performance
					# AUC (prediction probability)
					fpr_proba, tpr_proba, _ = roc_curve(y_test, pred_proba, pos_label=1)
					AUC_proba = auc(fpr_proba, tpr_proba)
					# AUPRC
					precision, recall, _ = precision_recall_curve(y_test, pred_proba, pos_label=1)
					AUPRC = auc(recall, precision)
					expected_AUPRC = list(y_test).count(1)/len(y_test)
					output['precisions'].append(','.join(map(str, precision)))
					output['recalls'].append(','.join(map(str, recall)))

					# final results
					output['train_dataset'].append(train_dataset)
					output['test_dataset'].append(test_dataset)
					output['iternum'].append(i)
					output['nGene'].append(nGene)
					output['qval'].append(qval)
					output['training_size'].append(training_size)
					output['training_sample_num'].append(len(y_train))					
					output['ML'].append(ML)
					output['test_type'].append(test_type)
					output['fpr_proba'].append(','.join(map(str, fpr_proba)))
					output['tpr_proba'].append(','.join(map(str, tpr_proba)))
					for metric, score, expected_score in zip(['AUC_proba', 'AUPRC'], [AUC_proba, AUPRC], [0.5, expected_AUPRC]):
						output[metric].append(score)
						output['expected_%s'%metric].append(expected_score)
						if metric == 'AUC_proba':
							AUCs.append(score)
						if metric == 'AUPRC':
							AUPRCs.append(score)

				print('\t%s -- average_AUC_proba : %s'%(test_type, np.mean(AUCs)))
				print('\t%s -- average_AUPRC : %s'%(test_type, np.mean(AUPRCs)))

output = pd.DataFrame(data=output, columns=output_col)
output.to_csv('%s/across_study_prediction_training_size_%s.txt'%(fo_dir, training_size), sep='\t', index=False)

print('Finished, %s'%time.ctime())
