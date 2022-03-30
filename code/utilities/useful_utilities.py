## USEFUL UTILITIES 
import pandas as pd
from collections import defaultdict
import numpy as np
import scipy.stats as stat
import time, os
import networkx as nx
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


## useful functions

#
def reactome_genes():
	output = defaultdict(list)
	output_list = []
	f = open('../../data/c2.all.v7.2.symbols.gmt','r')
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		if 'REACTOME' in line[0]:
			reactome = line[0]
			output_list.append(reactome)
			for i in range(2, len(line)):
				gene = line[i]
				output[reactome].append(gene)
	f.close()
	return output



