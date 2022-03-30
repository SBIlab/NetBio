## ML related utilities
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import numpy as np
import scipy.stats as stat
import time, os, math, random

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import log_loss, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.decomposition import PCA
import warnings
#exec(open('./useful_utilities.py').read())
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
loo = LeaveOneOut()

## expression scaler
def expression_StandardScaler(exp_df):
	'''
	Input : expression dataframe
	'''
	col1 = exp_df.columns[0]
	tmp = StandardScaler().fit_transform(exp_df.T.values[1:])
	new_tmp = defaultdict(list)
	new_tmp[col1] = exp_df[col1].tolist()
	for s_idx, sample in enumerate(exp_df.columns[1:]):
		new_tmp[sample] = tmp[s_idx]
	output = pd.DataFrame(data=new_tmp, columns=exp_df.columns)
	return output





## hyperparameter optimization
def ML_hyperparameters(ML, penalty='l2'):
	'''
	Input
	ML : 'SVC', 'RandomForest', 'LogisticRegression', 'RidgeClassifier'
	penalty : for LogisticRegression. 'none', 'l2'
	'''
	if ML == 'SVC':
		model = SVC()
		#param_grid = {'kernel':['rbf'], 'gamma':[0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100], 'C':[0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100], 'class_weight':['balanced'], 'probability':[True]}
		#param_grid = {'kernel':['linear', 'poly', 'sigmoid', 'rbf'], 'C':[0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100], 'class_weight':['balanced'], 'probability':[True]}
		param_grid = {'kernel':['linear'], 'C':[0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100], 'class_weight':['balanced'], 'probability':[True]}
	if ML == 'RandomForest':
		model = RFC()
		param_grid = {'n_estimators':[500, 1000], 'max_depth':[X_train.shape[0]], 'class_weight':['balanced']}
	if ML == 'LogisticRegression':
		model = LogisticRegression()
		if penalty == 'l2':
			param_grid = {'penalty':['l2'], 'max_iter':[1e10], 'solver':['lbfgs'], 'C':np.arange(0.1, 1, 0.1), 'class_weight':['balanced'] }
		if penalty == 'none':
			param_grid = {'penalty':['none'], 'max_iter':[1e10], 'class_weight':['balanced'] }

	if ML == 'RidgeClassifier':
		model = RidgeClassifier()
		param_grid = {'alpha':np.arange(0.1,1,0.1), 'class_weight':['balanced']}
	return model, param_grid




###========================================================================================================



## draw PCA
def draw_PCA(exp_array, samples, fi_dir, foName, scale_data=False):
	scaler = StandardScaler()
	if scale_data == True:
		scaled_array = scaler.fit_transform(exp_array)
	else:
		scaled_array = exp_array
	pca = PCA(n_components=2).fit(scaled_array)
	pc1_explained, pc2_explained = pca.explained_variance_ratio_
	pc = pca.transform(scaled_array)
	# tmp dataframe for PCA plot
	tmp = defaultdict(list)
	tmp_col = ['PC1', 'PC2', 'sample']
	for pc12, sample in zip(pc, samples):
		pc1, pc2 = pc12
		tmp['PC1'].append(pc1)
		tmp['PC2'].append(pc2)
		tmp['sample'].append(sample)
	tmp = pd.DataFrame(data=tmp, columns=tmp_col)

	fig, ax = plt.subplots(figsize=(10,10))
	#plt.figure(figsize=(10,10))
	ax.scatter(data=tmp, x='PC1', y='PC2')#, hue='sample')
	for i, txt in enumerate(tmp['sample'].tolist()):
		ax.annotate(txt, (tmp['PC1'].tolist()[i], tmp['PC2'].tolist()[i]), fontsize=8)
	ax.set_title(foName)
	ax.set_xlabel('PC1 (%s)'%pc1_explained)
	ax.set_ylabel('PC2 (%s)'%pc2_explained)
	#ax.tight_layout()
	plt.savefig('%s/PCA_%s.jpg'%(fi_dir, foName), format='jpg')
	plt.savefig('%s/PCA_%s.eps'%(fi_dir, foName), format='eps', dpi=300)
	#plt.show()
	plt.close()


def draw_two_cohorts_PCA(exp_array1, exp_array2, samples1, samples2, sample1_name, sample2_name, fi_dir, foName, scale_data=False):
	scaler = StandardScaler()
	exp1 = exp_array1
	exp2 = exp_array2
	if scale_data == True:
		exp1 = scaler.fit_transform(exp_array1)
		exp2 = scaler.fit_transform(exp_array2)
	exp = np.append(exp1, exp2, axis=0)
	samples = np.append(samples1, samples2)
	pca = PCA(n_components=2).fit(exp)
	pc1_explained, pc2_explained = pca.explained_variance_ratio_
	pc = pca.transform(exp)
	# tmp dataframe for PCA plot
	tmp = defaultdict(list)
	for pc12, sample in zip(pc, samples):
		pc1, pc2 = pc12
		tmp['PC1'].append(pc1)
		tmp['PC2'].append(pc2)
		if sample in samples1:
			tmp['sample'].append(sample1_name)
		if sample in samples2:
			tmp['sample'].append(sample2_name)
	tmp = pd.DataFrame(data=tmp)
	# draw figure
	f = plt.figure(figsize=(8,8))
	sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'sample', data = tmp)
	plt.title('%s\nPC1(var=%s)/PC2(var=%s)'%(foName, pc1_explained, pc2_explained))
	plt.tight_layout()
	#f.set_rasterized(True)

	plt.savefig('%s/%s.jpg'%(fi_dir, foName), format='jpg')
	plt.savefig('%s/%s.pdf'%(fi_dir, foName), format='pdf')
	plt.savefig('%s/%s.eps'%(fi_dir, foName), format='eps', dpi=300)

	plt.close()


