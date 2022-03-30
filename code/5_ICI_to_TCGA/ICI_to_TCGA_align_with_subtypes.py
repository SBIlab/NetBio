## align ICI response prediction results with known TCGA subtypes
import pandas as pd
from collections import defaultdict
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import seaborn as sns



## initialize
train_datasets = ['Gide', 'Liu']




## import subtypes
sdf = pd.read_csv('./TCGA_subtypes.csv')
sampleIDs = []
for sample in sdf['pan.samplesID'].tolist():
	sampleIDs.append(sample[:12])
sdf['sampleID'] = sampleIDs




## import prediction results
df = pd.read_csv('../../result/5_ICI_to_TCGA/ICI_to_TCGA_prediction.results.txt', sep='\t')




## merge dataframes
subtypes = []
for sample in df['sample'].tolist():
	tmp = sdf.loc[sdf['sampleID']==sample,:]
	subtype = np.nan
	if tmp.shape[0]>0:
		subtype = tmp['Subtype_mRNA'].tolist()[0]
	subtypes.append(subtype)
df['subtype_mRNA'] = subtypes
df2 = df.dropna()



## compare subtypes
def compare_subtypes():
	output_stat = defaultdict(list)


	for train_dataset in train_datasets:
		pred_scores = []
		tmpdf = defaultdict(list)

		tmp = df2.loc[df2['train_dataset']==train_dataset,:].loc[df2['subtype_mRNA']!='-',:].loc[df2['test_type']=='NetBio',:].dropna()
		subtypes = sorted(tmp['subtype_mRNA'].value_counts().index)
		xtick_labels = []

		for subtype in subtypes:
			pred_proba = tmp.loc[tmp['subtype_mRNA']==subtype,:]['predicted_response_proba'].tolist()
			pred_scores.append(pred_proba)
			xtick_labels.append('%s (%s)'%(subtype, len(pred_proba)))
			for prob in pred_proba:
				tmpdf['subtype'].append(subtype)
				tmpdf['pred proba'].append(prob)
		
		# statistical tests
		for s1_idx, s1 in enumerate(subtypes):
			for s2_idx, s2 in enumerate(subtypes):
				if s1_idx < s2_idx:
					_, pval = stat.mannwhitneyu(pred_scores[s1_idx], pred_scores[s2_idx])
					output_stat['train_dataset'].append(train_dataset)
					output_stat['subtype1'].append(s1)
					output_stat['subtype2'].append(s2)
					output_stat['mwu_pval'].append(pval)
		
		# boxplot
		tmpdf = pd.DataFrame(tmpdf)
		#sns.boxplot(x='subtype', y='pred proba', data=tmpdf)
		plt.boxplot(pred_scores)
		plt.xticks(np.arange(1, len(xtick_labels)+1), xtick_labels)
		plt.ylabel('Predicted probability of ICI response')
		plt.title(train_dataset)
		plt.tight_layout()
		plt.savefig('../../result/5_ICI_to_TCGA/align_with_TCGA_subtypes_%s.jpg'%train_dataset, format='jpg')
		plt.savefig('../../result/5_ICI_to_TCGA/align_with_TCGA_subtypes_%s.eps'%train_dataset, format='eps', dpi=300)
		plt.close()



	output_stat = pd.DataFrame(data=output_stat, columns=['train_dataset', 'subtype1', 'subtype2', 'mwu_pval'])
	output_stat.to_csv('../../result/5_ICI_to_TCGA/aligh_with_TCGA_subtypes_stats.txt', sep='\t', index=False)



## compare Gide and Liu among TCGA SKCM immune subtype patients
gdf = df2.loc[df2['train_dataset']=='Gide',:].loc[df2['test_dataset']=='TCGA.SKCM',:].loc[df2['test_type']=='NetBio',:].loc[df2['subtype_mRNA']=='immune',:]
ldf = df2.loc[df2['train_dataset']=='Liu',:].loc[df2['test_dataset']=='TCGA.SKCM',:].loc[df2['test_type']=='NetBio',:].loc[df2['subtype_mRNA']=='immune',:]

gdf = pd.DataFrame(data=gdf, columns=['sample', 'predicted_response_proba'])
ldf = pd.DataFrame(data=ldf, columns=['sample', 'predicted_response_proba'])

gdf = gdf.rename(columns={'predicted_response_proba':'Gide_score'})
ldf = ldf.rename(columns={'predicted_response_proba':'Liu_score'})

merged = pd.merge(gdf, ldf, on='sample', how='inner')

# stats and plots
coef, pval = stat.pearsonr(merged['Gide_score'].tolist(), merged['Liu_score'].tolist())
plt.figure(figsize=(6,6))
sns.regplot(x='Gide_score', y='Liu_score', data=merged)
plt.title('pearson coef = %.4f / pval = %.4f'%(coef, pval))
plt.tight_layout()
plt.savefig('../../result/5_ICI_to_TCGA/Gide_vs_Liu_immune_subtype.jpg', format='jpg')
plt.savefig('../../result/5_ICI_to_TCGA/Gide_vs_Liu_immune_subtype.eps', format='eps', dpi=300)
plt.close()


