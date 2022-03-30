## codes for Deep Neural Network (DNN)
## DNN architecture from Sakellaropoulos et al, 2019
## This code includes:
#		Deep Neural Network
#		5Fold cross-validation-based hyper-parameter optimization
# batch size = 16
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os, time, random
from collections import defaultdict



# define class
class TensorData(Dataset):
	def __init__(self, x_data, y_data):
		self.x_data = torch.FloatTensor(x_data)
		self.y_data = torch.FloatTensor(y_data)
		self.len = self.y_data.shape[0]
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]
	def __len__(self):
		return self.len



class DeepNN_v1(nn.Module):
	def __init__(self, X_dim, i_dropout_rate, h_dropout_rate):
		super().__init__()
		#
		self.v1_layer1 = nn.Linear(X_dim, 512, bias=True)
		self.v1_layer2 = nn.Linear(512, 1, bias=True)
		# 
		self.i_dropout = nn.Dropout(i_dropout_rate)
		self.h_dropout = nn.Dropout(h_dropout_rate)

	def forward(self, x):
		x = self.i_dropout(torch.tanh(self.v1_layer1(x)))
		x = torch.sigmoid(self.v1_layer2(x))
		return x


class DeepNN_v2(nn.Module):
	def __init__(self, X_dim, i_dropout_rate, h_dropout_rate):
		super().__init__()
		#
		self.v2_layer1 = nn.Linear(X_dim, 256, bias=True)
		self.v2_layer2 = nn.Linear(256, 256, bias=True)
		self.v2_layer3 = nn.Linear(256, 1, bias=True)
		# 
		self.i_dropout = nn.Dropout(i_dropout_rate)
		self.h_dropout = nn.Dropout(h_dropout_rate)

	def forward(self, x):
		x = self.i_dropout(torch.tanh(self.v2_layer1(x)))
		x = self.h_dropout(torch.tanh(self.v2_layer2(x)))
		x = torch.sigmoid(self.v2_layer3(x))
		return x


class DeepNN_v3(nn.Module):
	def __init__(self, X_dim, i_dropout_rate, h_dropout_rate):
		super().__init__()
		#
		self.v3_layer1 = nn.Linear(X_dim, 128, bias=True)
		self.v3_layer2 = nn.Linear(128, 128, bias=True)
		self.v3_layer3 = nn.Linear(128, 128, bias=True)
		self.v3_layer4 = nn.Linear(128, 1, bias=True)
		# 
		self.i_dropout = nn.Dropout(i_dropout_rate)
		self.h_dropout = nn.Dropout(h_dropout_rate)
	
	def forward(self, x):
		x = self.i_dropout(torch.tanh(self.v3_layer1(x)))
		x = self.h_dropout(torch.tanh(self.v3_layer2(x)))
		x = self.h_dropout(torch.tanh(self.v3_layer3(x)))
		x = torch.sigmoid(self.v3_layer4(x))
		return x


class DeepNN_v4(nn.Module):
	def __init__(self, X_dim, i_dropout_rate, h_dropout_rate):
		super().__init__()
		#
		self.v4_layer1 = nn.Linear(X_dim, 128, bias=True)
		self.v4_layer2 = nn.Linear(128, 128, bias=True)
		self.v4_layer3 = nn.Linear(128, 64, bias=True)
		self.v4_layer4 = nn.Linear(64, 1, bias=True)
		# 
		self.i_dropout = nn.Dropout(i_dropout_rate)
		self.h_dropout = nn.Dropout(h_dropout_rate)
	
	def forward(self, x):
		x = self.i_dropout(torch.tanh(self.v4_layer1(x)))
		#x = self.h_dropout(torch.tanh(self.v4_layer2(x)))
		x = self.h_dropout(torch.relu(self.v4_layer2(x)))
		x = self.h_dropout(torch.relu(self.v4_layer3(x)))
		x = torch.sigmoid(self.v4_layer4(x))
		return x


# define functions
def return_dataloader(X, y):
	# TensorData 
	dataset = TensorData(X, y)
	# DataLoader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
	return dataloader

def binary_classification(predictions, cutoff=0.5):
	new_predictions = []
	for pred in predictions:
		p = pred[0]
		if p >= cutoff:
			new_predictions.append([1])
		else:
			new_predictions.append([0])
	return np.array(new_predictions)




# train_DNN
def train_DNN(X_train, y_train, X_test, y_test, hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty):
	# trainloader, testloader
	trainloader = return_dataloader(X_train, y_train)
	testloader = return_dataloader(X_test, y_test)
	
	# cuda
	torch.cuda.set_device(1)

	# train DNN
	model = []
	if (hidden_layer == 'v1') or (hidden_layer == '1x512'):
		model = DeepNN_v1(X_dim=X_train.shape[1], i_dropout_rate=i_dropout_rate, h_dropout_rate=h_dropout_rate).train()
	if (hidden_layer == 'v2') or (hidden_layer == '2x256'):
		model = DeepNN_v2(X_dim=X_train.shape[1], i_dropout_rate=i_dropout_rate, h_dropout_rate=h_dropout_rate).train()
	if (hidden_layer == 'v3') or (hidden_layer == '3x128'):
		model = DeepNN_v3(X_dim=X_train.shape[1], i_dropout_rate=i_dropout_rate, h_dropout_rate=h_dropout_rate).train()
	if (hidden_layer == 'v4') or (hidden_layer == '1x128'):
		model = DeepNN_v4(X_dim=X_train.shape[1], i_dropout_rate=i_dropout_rate, h_dropout_rate=h_dropout_rate).train()
	model = model.cuda()
	
	#criterion = nn.MSELoss().cuda()
	criterion = nn.BCELoss().cuda()
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_penalty)

	loss_ = []
	n = len(trainloader)

	for e in range(epoch):
		running_loss = 0
		for i, data in enumerate(trainloader, 0):
			inputs, values = data
			inputs = inputs.cuda(); values = values.cuda()
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, values.reshape(-1,1))
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		loss_.append(running_loss/n)
	
	# evaluate
	predictions = torch.tensor([], dtype=torch.float)
	actual = torch.tensor([], dtype=torch.float)

	with torch.no_grad():
		model.eval()

		for data in testloader:
			inputs, values = data
			inputs = inputs.cuda(); values = values.cuda()
			outputs = model(inputs).cpu()

			predictions = torch.cat((predictions, outputs),0)
			values = values.cpu()
			actual = torch.cat((actual, values),0)
	predictions = predictions.numpy()
	actual = actual.numpy()
	return predictions, actual, loss_



# train DNN with hyperparameter optimization (grid search)
def train_DNN_with_optimization(X_train, y_train, X_test, y_test, hidden_layers, i_dropout_rates, h_dropout_rates, epochs, l2_penalties, CV_fold=5, scoring='roc_auc'):
	tmp = defaultdict(list)
	tmp2 = defaultdict(list)
	# hyper-parameter gridsearch
	#print('starting hyper-parameter gridsearch, %s'%time.ctime())
	kfold = KFold(n_splits=CV_fold)
	for train_index, test_index in kfold.split(X_train):
		X2_train, y2_train = X_train[train_index], y_train[train_index]
		X2_test, y2_test = X_train[test_index], y_train[test_index]
		# 'continue' if y2_train has only one class
		if (list(y_train.reshape(-1)).count(1) == len(y_train)) or (list(y_train.reshape(-1)).count(1) == 0):
			continue
		# hyper-parameters
		for hidden_layer in hidden_layers:
			for i_dropout_rate in i_dropout_rates:
				for h_dropout_rate in h_dropout_rates:
					for epoch in epochs:
						for l2_penalty in l2_penalties:
							predictions, actual, loss_ = train_DNN(X2_train, y2_train, X2_test, y2_test, hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty)
							if scoring == 'roc_auc':
								fpr, tpr, _ = roc_curve(actual, predictions, pos_label=1)
								score = auc(fpr, tpr)
								tmp['score'].append(score)
							for key, value in zip(['hidden_layer', 'i_dropout_rate', 'h_dropout_rate', 'epoch', 'l2_penalty'], [hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty]):
								tmp[key].append(value)
	# tmp2
	tmp = pd.DataFrame(data=tmp).dropna()
	for hidden_layer in hidden_layers:
		for i_dropout_rate in i_dropout_rates:
			for h_dropout_rate in h_dropout_rates:
				for epoch in epochs:
					for l2_penalty in l2_penalties:
						for key, value in zip(['hidden_layer', 'i_dropout_rate', 'h_dropout_rate', 'epoch', 'l2_penalty'], [hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty]):
							tmp2[key].append(value)
						scores = tmp.loc[tmp['hidden_layer']==hidden_layer,:].loc[tmp['i_dropout_rate']==i_dropout_rate,:].loc[tmp['h_dropout_rate']==h_dropout_rate,:].loc[tmp['epoch']==epoch,:].loc[tmp['l2_penalty']==l2_penalty,:]['score'].tolist()
						tmp2['score'].append(np.mean(scores))
	tmp2 = pd.DataFrame(tmp2)
	if scoring == 'roc_auc':
		tmp2 = tmp2.sort_values(by='score', ascending=False)
		hidden_layer = tmp2['hidden_layer'].tolist()[0]
		i_dropout_rate = tmp2['i_dropout_rate'].tolist()[0]
		h_dropout_rate = tmp2['h_dropout_rate'].tolist()[0]
		epoch = tmp2['epoch'].tolist()[0]
		l2_penalty = tmp2['l2_penalty'].tolist()[0]
	# train DNN
	predictions, actual, loss_ = train_DNN(X_train, y_train, X_test, y_test, hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty)
	binary_predictions = binary_classification(predictions)
	return predictions, actual, binary_predictions, loss_, tmp, tmp2
	



# train DNN with hyperparameter optimization (test randomly selected hyperparameters)
def train_DNN_with_optimization_and_random_select_hyperparam(X_train, y_train, X_test, y_test, hidden_layers, i_dropout_rates, h_dropout_rates, epochs, l2_penalties, CV_fold=5, scoring='roc_auc', num_hyperparam_sets = 10):
	tmp = defaultdict(list)
	tmp2 = defaultdict(list)
	# randomly selecthyper-parameters
	selected_hyperparams = []
	total_hyperparams = len(hidden_layers) * len(i_dropout_rates) * len(h_dropout_rates) * len(epochs) * len(l2_penalties)
	for i in range(total_hyperparams):
		hl = random.sample(hidden_layers, 1)[0]
		idr = random.sample(i_dropout_rates, 1)[0]
		hdr = random.sample(h_dropout_rates, 1)[0]
		ep = random.sample(epochs, 1)[0]
		l2p = random.sample(l2_penalties, 1)[0]
		s_list = (hl, idr, hdr, ep, l2p)
		if not s_list in selected_hyperparams:
			selected_hyperparams.append(s_list)
		if len(selected_hyperparams) == num_hyperparam_sets:
			break

	# KFold cross validation
	kfold = KFold(n_splits=CV_fold)
	for train_index, test_index in kfold.split(X_train):
		X2_train, y2_train = X_train[train_index], y_train[train_index]
		X2_test, y2_test = X_train[test_index], y_train[test_index]
		# 'continue' if y2_train has only one class
		if (list(y_train.reshape(-1)).count(1) == len(y_train)) or (list(y_train.reshape(-1)).count(1) == 0):
			continue
		# hyper-parameters
		for s_list in selected_hyperparams:
			hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty = s_list
			predictions, actual, loss_ = train_DNN(X2_train, y2_train, X2_test, y2_test, hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty)
			if scoring == 'roc_auc':
				fpr, tpr, _ = roc_curve(actual, predictions, pos_label=1)
				score = auc(fpr, tpr)
				tmp['score'].append(score)
			for key, value in zip(['hidden_layer', 'i_dropout_rate', 'h_dropout_rate', 'epoch', 'l2_penalty'], [hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty]):
				tmp[key].append(value)
	# tmp2
	tmp = pd.DataFrame(data=tmp).dropna()
	for s_list in selected_hyperparams:
		hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty = s_list
		for key, value in zip(['hidden_layer', 'i_dropout_rate', 'h_dropout_rate', 'epoch', 'l2_penalty'], [hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty]):
			tmp2[key].append(value)
		scores = tmp.loc[tmp['hidden_layer']==hidden_layer,:].loc[tmp['i_dropout_rate']==i_dropout_rate,:].loc[tmp['h_dropout_rate']==h_dropout_rate,:].loc[tmp['epoch']==epoch,:].loc[tmp['l2_penalty']==l2_penalty,:]['score'].tolist()
		tmp2['score'].append(np.mean(scores))
	tmp2 = pd.DataFrame(tmp2)
	if scoring == 'roc_auc':
		tmp2 = tmp2.sort_values(by='score', ascending=False)
		hidden_layer = tmp2['hidden_layer'].tolist()[0]
		i_dropout_rate = tmp2['i_dropout_rate'].tolist()[0]
		h_dropout_rate = tmp2['h_dropout_rate'].tolist()[0]
		epoch = tmp2['epoch'].tolist()[0]
		l2_penalty = tmp2['l2_penalty'].tolist()[0]
	# train DNN
	predictions, actual, loss_ = train_DNN(X_train, y_train, X_test, y_test, hidden_layer, i_dropout_rate, h_dropout_rate, epoch, l2_penalty)
	binary_predictions = binary_classification(predictions)
	return predictions, actual, binary_predictions, loss_, tmp, tmp2
