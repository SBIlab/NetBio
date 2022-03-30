## across study prediction using deep neural network (DNN)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut
exec(open('../utilities/parse_patient_data.py').read())
exec(open('../utilities/ML.py').read())
exec(open('../utilities/DNN.py').read())
exec(open('../utilities/useful_utilities.py').read())




## Initialize
studies = ['Prat_MELANOMA', 'Riaz_pre', 'Auslander', 'Huang_pre']
ML = 'DNN'; test_type = 'DNN'
num_hyperparam_sets = 10
hidden_layers = ['v1', 'v2', 'v3']
i_dropout_rates = [0, 0.1, 0.5]
h_dropout_rates = [0.1, 0.3, 0.5]
epochs = [10, 50, 100, 200]
l2_penalties = [1e-2, 1e-4]

training_size = 0.8
iternum = 100


cohort_targets = {'PD1_CTLA4':['Gide'], 'PD1':['Liu']}
train_datasets = []
for key in list(cohort_targets.keys()):
	for value in cohort_targets[key]:
		train_datasets.append(value)



## Output dataframe
output = defaultdict(list)
output_col = ['train_dataset', 'test_dataset', 'iternum', 'training_size', 'training_sample_num', 'nGene', 'qval', 'ML', 'test_type', 'AUC_proba', 'fpr_proba', 'tpr_proba', 'AUPRC', 'expected_AUPRC', 'precisions', 'recalls']


output_dir = '../../result/2_cross_study_prediction/DNN'


## Run DNN
for train_idx, train_dataset in enumerate(train_datasets):
	for test_idx, test_dataset in enumerate(studies):


		## train dataset targets
		for key in list(cohort_targets.keys()):
			if train_dataset in cohort_targets[key]:
				train_targets = key
				target = key
				break

		### load datasets
		if ('Riaz' in train_dataset) or ('Huang' in train_dataset):
			train_samples, train_edf, _, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], drug_treatment=train_dataset.split('_')[1])
		elif 'Prat' in train_dataset:
			train_samples, train_edf, _, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], Prat_cancer_type=train_dataset.split('_')[1])
		else:
			train_samples, train_edf, _, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset)
		if ('Riaz' in test_dataset) or ('Huang' in test_dataset):
			test_samples, test_edf, _, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset.split('_')[0], drug_treatment=test_dataset.split('_')[1])
		elif 'Prat' in test_dataset:
			test_samples, test_edf, _, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset.split('_')[0], Prat_cancer_type=test_dataset.split('_')[1])
		else:
			test_samples, test_edf, _, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset)

		### requirements
		if train_dataset == test_dataset:
			continue
		if ('Riaz' in train_dataset) and ('Riaz' in test_dataset):
			continue
		if (len(train_samples) < 30) or (train_responses.tolist().count(1) < 10) or (train_responses.tolist().count(0) < 10):
			continue

		print('\n\n#----------------------------------------')
		print('training: %s, test: %s, %s'%(train_dataset, test_dataset, time.ctime()))
		print('test data --> responder : %s / non-responder : %s'%(list(test_responses).count(1), list(test_responses).count(0)))
		


		### data cleanup: match genes and pathways between cohorts
		common_genes = list(set(train_edf['genes'].tolist()) & set(test_edf['genes'].tolist()))
		train_edf = train_edf.loc[train_edf['genes'].isin(common_genes),:].sort_values(by='genes')
		test_edf = test_edf.loc[test_edf['genes'].isin(common_genes),:].sort_values(by='genes')

		### data cleanup: expression standardization
		edf1, edf2 = train_edf, test_edf
		train_edf = expression_StandardScaler(train_edf)
		test_edf = expression_StandardScaler(test_edf)

		### train test expression data
		tmp_X_train, X_test = train_edf.T.values[1:].astype(np.float), train_edf.T.values[1:].astype(np.float)
		tmp_y_train, y_test = train_responses.astype(np.float), test_responses.astype(np.float)
		if tmp_X_train.shape[1] * tmp_X_train.shape[0] * len(tmp_y_train) * len(y_test) == 0:
			continue

		
		### Measure prediction performances
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
			predictions, actual, binary_predictions, loss_, hyperparam_tmp, hyperparam_tmp2 = train_DNN_with_optimization_and_random_select_hyperparam(X_train, y_train, X_test, y_test, hidden_layers, i_dropout_rates, h_dropout_rates, epochs, l2_penalties, num_hyperparam_sets=num_hyperparam_sets)
			pred_proba = predictions.reshape(-1)
			pred_status = binary_predictions.reshape(-1)


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
					if (i == 1) or (i%10==0):
						print('\t\t%s / %s, avgerage AUC : %s'%(i, iternum, np.mean(AUCs)))
				if metric == 'AUPRC':
					AUPRCs.append(score)
					if (i == 1) or (i%10==0):
						print('\t\t%s / %s, avgerage AUPRC : %s'%(i, iternum, np.mean(AUPRCs)))
		

		print('\t%s -- average_AUC_proba : %s'%(test_type, np.mean(AUCs)))
		print('\t%s -- average_AUPRC : %s'%(test_type, np.mean(AUPRCs)))
		

output = pd.DataFrame(data=output, columns=output_col)
output.to_csv('%s/across_study_performance_reduced_training_DNN.txt'%output_dir, sep='\t', index=False)

print('Finished, %s'%time.ctime())

