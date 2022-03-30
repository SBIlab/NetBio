## compare LOOCV prediction performances and overall survival
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import os, time
from collections import defaultdict
import lifelines; from lifelines.statistics import logrank_test; from lifelines import KaplanMeierFitter
exec(open('../utilities/parse_patient_data.py').read())

## initialize
ML = 'LogisticRegression'


# dataset to exclude
datasets_to_exclude = []


# data directory
data_dir = '../../result/1_within_study_prediction' # '/home/junghokong/BLCA_cisplatin_immunotherapy/result/3_immunotherapy_prediction/LOOCV'

# output directory
fo_dir = data_dir
if os.path.isdir('%s/survival_plot'%(fo_dir)) == False:
	os.mkdir('%s/survival_plot'%fo_dir)
fo_dir = '%s/survival_plot'%fo_dir


# import data and plot overall survival
for fi in os.listdir(data_dir):
	if ('PredictedResponse.txt' in fi) and ('LOOCV_%s'%ML in fi):
		df = pd.read_csv('%s/%s'%(data_dir, fi), sep='\t')
		# study
		studies = list(set(df['study'].tolist()))
		for study in studies:
			for survival_type in ['os', 'pfs']:
				try:
					_, pdf = parse_immunotherapy_survival(study, survival_type=survival_type)
				except: continue
				# dataset to exclude
				if study in datasets_to_exclude:
					continue
				if pdf.shape[0] == 0:
					continue
				# test start
				print('testing %s-%s, %s'%(study, survival_type, time.ctime()))



				# merged
				merged = pd.merge(df, pdf, on='sample', how='inner')
				# make os plot for each test_type
				for test_type in list(set(merged['test_type'].tolist())):
					tmp = merged.loc[merged['test_type']==test_type,:]
					rdf = tmp.loc[tmp['predicted_response']==1,:]
					nrdf = tmp.loc[tmp['predicted_response']==0,:]
					# stats
					results = logrank_test(rdf[survival_type].tolist(), nrdf[survival_type].tolist(), 
							event_observed_A=rdf['%s_status'%survival_type].tolist(), event_observed_B=nrdf['%s_status'%survival_type].tolist())
					pvalue = results.p_value
					kmf = KaplanMeierFitter()
					kmf.fit(rdf[survival_type].tolist(), rdf['%s_status'%survival_type].tolist())
					r_median = kmf.median_survival_time_ # median survival for predicted responders
					kmf = KaplanMeierFitter()
					kmf.fit(nrdf[survival_type].tolist(), nrdf['%s_status'%survival_type].tolist())
					nr_median = kmf.median_survival_time_ # median survival for predicted nonresponders

					# draw survival plot
					f = plt.figure(figsize=(8,8))
					ax = f.add_subplot(1,1,1)
					c1 = KaplanMeierFitter()
					ax = c1.fit(rdf[survival_type].tolist(), rdf['%s_status'%survival_type].tolist(), label='Predicted Responder (N=%s)'%(len(rdf[survival_type].tolist()))).plot(ax=ax, ci_show=True, color='r')
					c2 = KaplanMeierFitter()
					ax = c2.fit(nrdf[survival_type].tolist(), nrdf['%s_status'%survival_type].tolist(), label='Predicted NonResponder (N=%s)'%(len(nrdf[survival_type].tolist()))).plot(ax=ax, ci_show=True, color='b')
					plt.xlabel('Survival (%s)'%tmp['%s_type'%survival_type].tolist()[0])
					plt.ylabel('Percent Survival')
					ymin, ymax = 0, 1.1
					plt.xlim(0)
					plt.ylim(ymin, ymax)
					plt.title('%s (%s) / %s\npvalue = %s\npred(responder) median = %s\npred(nonresponder) = %s'%(study, survival_type, test_type, pvalue, r_median, nr_median))
					plt.tight_layout()
					#ax.set_rasterized(True)
					# make output directory
					if os.path.isdir('%s/%s'%(fo_dir, fi.replace('.txt',''))) == False:
						os.mkdir('%s/%s'%(fo_dir, fi.replace('.txt','')))
					plt_dir = '%s/%s'%(fo_dir, fi.replace('.txt',''))
					if os.path.isdir('%s/%s'%(plt_dir, study)) == False:
						os.mkdir('%s/%s'%(plt_dir, study))
					plt_dir = '%s/%s'%(plt_dir, study)
					# save figure
					plt.savefig('%s/%s_%s_%s.jpg'%(plt_dir, survival_type.upper(), study, test_type), format='jpg')
					plt.savefig('%s/%s_%s_%s.eps'%(plt_dir, survival_type.upper(), study, test_type), format='eps', dpi=300)
					plt.close()
