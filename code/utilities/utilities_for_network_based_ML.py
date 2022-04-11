## codes for network-based feature selection and ML prediction
import pandas as pd
from collections import defaultdict
import scipy.stats as stat
import numpy as np
import time, os
from statsmodels.stats.multitest import multipletests
exec(open('./useful_utilities.py').read())



def return_proximal_pathways(edf, seed, nGene, adj_pval):
	'''
	Inputs
	edf: gene expression dataframe
	seed: seed genes for network expansion. MUST BE PROVIDED IN A STRING FORMAT!!
	nGene: nGene
	adj_pval: adjusted pvalue cutoff
	'''
	reactome = reactome_genes()

	# results from gene expansion by network propagation
	fi_dir = '../../result/0_data_collection_and_preprocessing' #'/home/junghokong/BLCA_cisplatin_immunotherapy/result/network_propagated_scores/immune_biomarkers'
	bdf = pd.read_csv('%s/%s.txt'%(fi_dir, seed), sep='\t')
	bdf = bdf.dropna(subset=['gene_id'])
	b_genes = []
	for idx, gene in enumerate(bdf.sort_values(by=['propagate_score'], ascending=False)['gene_id'].tolist()):
		if gene in edf['genes'].tolist():
			if not gene in b_genes:
				b_genes.append(gene)
			if len(set(b_genes)) >= nGene:
				break
	
	# LCC function enrichment
	tmp_hypergeom = defaultdict(list)
	pvalues, adj_pvalues = [], []
	for pw in reactome.keys():
		pw_genes = list(set(reactome[pw]) & set(edf['genes'].tolist()))
		M = len(edf['genes'].tolist())
		n = len(pw_genes)
		N = len(set(b_genes))
		k = len(set(pw_genes) & set(b_genes))
		p = stat.hypergeom.sf(k-1, M, n, N)
		tmp_hypergeom['pw'].append(pw)
		tmp_hypergeom['p'].append(p)
		pvalues.append(p)
		_, adj_pvalues, _, _ = multipletests(pvalues)
	tmp_hypergeom['adj_p'] = adj_pvalues
	tmp_hypergeom = pd.DataFrame(tmp_hypergeom).sort_values(by=['adj_p'])
	proximal_pathways = tmp_hypergeom.loc[tmp_hypergeom['adj_p']<=adj_pval,:]['pw'].tolist()
	return proximal_pathways
