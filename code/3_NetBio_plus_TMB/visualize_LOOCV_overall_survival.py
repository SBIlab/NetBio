## compare LOOCV prediction performances and overall survival
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import os, time
from collections import defaultdict
import lifelines; from lifelines.statistics import logrank_test; from lifelines import KaplanMeierFitter
exec(open('../utilities/parse_patient_data.py').read())

## Initialize
nGene = 200
qval = 0.01
study = 'IMvigor210' # 'Liu', 'IMvigor210'
compare1 = ['TMB']
if study == 'IMvigor210':
	compare2 = ['TMB.plus.PD-L1_NetBio']
if study == 'Liu':
	compare2 = ['TMB.plus.PD1_NetBio']




# data directory
data_dir = '../../result/3_NetBio_plus_TMB'

# output directory
fo_dir = data_dir
if os.path.isdir('%s/survival_plot'%(fo_dir)) == False:
	os.mkdir('%s/survival_plot'%fo_dir)
fo_dir = '%s/survival_plot'%fo_dir


# output stats
output = defaultdict(list)
output_col = ['dataset', 'group1', 'group2', 'group1_one_year', 'group1_two_year', 'group1_three_year', 'group1_five_year', 'group2_one_year', 'group2_two_year', 'group2_three_year', 'group2_five_year', 'logrank_pvalue']


# import data and plot overall survival
for fi in os.listdir(data_dir):
	if 'PredictedResponse.txt' in fi:
		df = pd.read_csv('%s/%s'%(data_dir, fi), sep='\t')
		df = df.loc[df['nGene']==nGene,:].loc[df['qval']==qval,:]
		# study
		try:
			_, pdf = parse_immunotherapy_survival(study)
		except: continue
		# test start
		print('testing %s, %s'%(study, time.ctime()))

		# merged
		merged = pd.merge(df, pdf, on='sample', how='inner')
		
		
		# make os plot for each test_type
		for test_type in list(set(merged['test_type'].tolist())):
			try:
				tmp = merged.loc[merged['test_type']==test_type,:]
				rdf = tmp.loc[tmp['predicted_response']==1,:]
				nrdf = tmp.loc[tmp['predicted_response']==0,:]
				# stats
				results = logrank_test(rdf['os'].tolist(), nrdf['os'].tolist(), 
						event_observed_A=rdf['os_status'].tolist(), event_observed_B=nrdf['os_status'].tolist())
				pvalue = results.p_value
				kmf = KaplanMeierFitter()
				kmf.fit(rdf['os'].tolist(), rdf['os_status'].tolist())
				r_median = kmf.median_survival_time_ # median survival for predicted responders
				year_proba = []
				for months, years in zip([12, 24, 36, 60], [1,2,3,5]):
					proba = kmf.predict(months)
					year_proba.append(proba)
				kmf = KaplanMeierFitter()
				kmf.fit(nrdf['os'].tolist(), nrdf['os_status'].tolist())
				nr_median = kmf.median_survival_time_ # median survival for predicted nonresponders
				for months, years in zip([12, 24, 36, 60], [1,2,3,5]):
					proba = kmf.predict(months)
					year_proba.append(proba)
				# output
				scores = np.append(year_proba, [pvalue])
				for output_key, output_value in zip(output_col, np.append([study, '%s_responder'%test_type, '%s_nonresponder'%test_type], scores)):
					output[output_key].append(output_value)

				# draw survival plot
				f = plt.figure(figsize=(8,8))
				ax = f.add_subplot(1,1,1)
				c1 = KaplanMeierFitter()
				ax = c1.fit(rdf['os'].tolist(), rdf['os_status'].tolist(), label='Predicted Responder (N=%s)'%(len(rdf['os'].tolist()))).plot(ax=ax, ci_show=True, color='r')
				c2 = KaplanMeierFitter()
				ax = c2.fit(nrdf['os'].tolist(), nrdf['os_status'].tolist(), label='Predicted NonResponder (N=%s)'%(len(nrdf['os'].tolist()))).plot(ax=ax, ci_show=True, color='b')
				plt.xlabel('Survival (%s)'%tmp['os_type'].tolist()[0])
				plt.ylabel('Percent Survival')
				ymin, ymax = 0, 1.1
				plt.xlim(0)
				plt.ylim(ymin, ymax)
				plt.title('%s / %s\npvalue = %s\npred(responder) median = %s\npred(nonresponder) = %s'%(study, test_type, pvalue, r_median, nr_median))
				if study == 'IMvigor210':
					plt.plot([12, 12], [0,1], 'k--')
				plt.tight_layout()
				plt.savefig('%s/LOOCV_%s_%s.jpg'%(fo_dir, study, test_type), format='jpg')
				plt.savefig('%s/LOOCV_%s_%s.eps'%(fo_dir, study, test_type), format='eps', dpi=300)
				plt.close()
			except:
				pass



		## os plot for original VS reclassified group
		original_dic = defaultdict(list)
		reclassified_dic = defaultdict(list)
		tmp1 = merged.loc[merged['study']==study,:].loc[merged['nGene']==nGene,:].loc[merged['qval']==qval,:].loc[merged['test_type'].isin(compare1),:]
		tmp2 = merged.loc[merged['study']==study,:].loc[merged['nGene']==nGene,:].loc[merged['qval']==qval,:].loc[merged['test_type'].isin(compare2),:]
		for sample, response1 in zip(tmp1['sample'].tolist(), tmp1['predicted_response']):
			response2 = tmp2.loc[tmp2['sample']==sample,:]['predicted_response'].tolist()[0]
			if response1 == 1:
				R1 = 'R'
			else:
				R1 = 'NR'
			if response2 == 1:
				R2 = 'R'
			else:
				R2 = 'NR'
			original_dic[R1].append(sample)
			if R1 != R2:
				reclassified_dic['%s2%s'%(R1, R2)].append(sample)
		# samples
		original_R, original_NR = original_dic['R'], original_dic['NR']
		R2NR, NR2R = reclassified_dic['R2NR'], reclassified_dic['NR2R']
		original_R_exc, original_NR_exc = list(set(original_R)-set(R2NR)), list(set(original_NR)-set(NR2R))
		for oR, oNR, key in zip([original_R_exc], [original_NR_exc], ['exclusive']):			
			for o_samples, o_response, reclassified_key in zip([oR, oNR], ['R', 'NR'], ['R2NR', 'NR2R']):
				original_key = '%s %s Pred %s'%('_'.join(map(str, compare1)), key, o_response)
				if 'Pred R' in original_key:
					original_key = 'R2R'
				if 'Pred NR' in original_key:
					original_key = 'NR2NR'

				original_df = pdf.loc[pdf['sample'].isin(o_samples),:]
				reclass_df = pdf.loc[pdf['sample'].isin(reclassified_dic[reclassified_key]),:]
				
				# stats
				results = logrank_test(original_df['os'].tolist(), reclass_df['os'].tolist(), 
						event_observed_A=original_df['os_status'].tolist(), event_observed_B=reclass_df['os_status'].tolist())
				pvalue = results.p_value
				kmf = KaplanMeierFitter()
				kmf.fit(original_df['os'].tolist(), original_df['os_status'].tolist())
				original_median = kmf.median_survival_time_ # median survival
				year_proba = []
				for months, years in zip([12, 24, 36, 60], [1,2,3,5]):
					proba = kmf.predict(months)
					year_proba.append(proba)
				kmf = KaplanMeierFitter()
				kmf.fit(reclass_df['os'].tolist(), reclass_df['os_status'].tolist())
				reclassified_median = kmf.median_survival_time_ # median survival
				for months, years in zip([12, 24, 36, 60], [1,2,3,5]):
					proba = kmf.predict(months)
					year_proba.append(proba)
				# output
				scores = np.append(year_proba, [pvalue])
				for output_key, output_value in zip(output_col, np.append([study, original_key, reclassified_key], scores)):
					output[output_key].append(output_value)
				

				# draw survival plot
				f = plt.figure(figsize=(8,8))
				ax = f.add_subplot(1,1,1)
				c1 = KaplanMeierFitter()
				ax = c1.fit(original_df['os'].tolist(), original_df['os_status'].tolist(), label='%s (N=%s)'%(original_key, len(original_df['os'].tolist()))).plot(ax=ax, ci_show=True, color='r')
				c2 = KaplanMeierFitter()
				ax = c2.fit(reclass_df['os'].tolist(), reclass_df['os_status'].tolist(), label='%s (N=%s)'%(reclassified_key, len(reclass_df['os'].tolist()))).plot(ax=ax, ci_show=True, color='b')
				plt.xlabel('Survival (%s)'%pdf['os_type'].tolist()[0])
				plt.ylabel('Percent Survival')
				ymin, ymax = 0, 1.1
				plt.xlim(0)
				plt.ylim(ymin, ymax)
				plt.title('%s\npvalue = %s\n%s median = %s\n%s = %s'%(study, pvalue, original_key, original_median, reclassified_key, reclassified_median))
				plt.tight_layout()
				plt.savefig('%s/%s_%s_vs_%s.jpg'%(fo_dir, study, original_key, reclassified_key), format='jpg')
				plt.savefig('%s/%s_%s_vs_%s.eps'%(fo_dir, study, original_key, reclassified_key), format='eps', dpi=300)
				plt.close()

output = pd.DataFrame(data=output, columns=output_col)
output.to_csv('%s/stats.txt'%fo_dir, sep='\t', index=False)
