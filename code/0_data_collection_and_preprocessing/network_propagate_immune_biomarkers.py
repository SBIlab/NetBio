## network propagation for immunotherapy biomarkers
import numpy as np
from collections import defaultdict
import scipy.stats as stat
import pandas as pd
import time, os, random
import networkx as nx
from networkx.algorithms.link_analysis import pagerank
exec(open('../utilities/network_utilities_ver3.py').read())
exec(open('../utilities/useful_utilities.py').read())
start = time.ctime()


## Initialize
string_cutoff = 700
fo_dir = '../../result'

## Prepare data

#  Load immunotherapy biomarkers
ib = pd.read_csv('../../data/Marker_summary.txt', sep='\t')


#  Construct Network (STRING network v11)
print('constructing STRING PPI network, %s'%time.ctime())
tmp_G = nx.Graph()
annotation = pd.read_csv('../../data/9606.protein.aliases.v11.0.txt', sep='\t') # STRING network ensembl-geneID mapping
net = pd.read_csv('../../data/9606.protein.links.v11.0.txt', sep=' ') # STRING network
nodes1 = net.values[:,0]
nodes2 = net.values[:,1]
scores = net.values[:,2]
for n1, n2, score in zip(nodes1, nodes2, scores):
        if score >= string_cutoff:
                tmp_G.add_edge(n1, n2)
LCC_genes = max(nx.connected_components(tmp_G), key=len)
G = tmp_G.subgraph(LCC_genes) ## Largest Connected Componenets
network_nodes = G.nodes()
network_edges = G.edges()
print('network nodes: %s'%len(network_nodes))
print('network edges: %s'%len(network_edges))
print('\n')


#  annotation
print('make annotation dictionary, ', time.ctime())
anno = pd.DataFrame(data=annotation.loc[annotation['source'].str.contains('HUGO', na=False),:])
anno_dic = defaultdict(list) # { ensp : [ genes ] }
for ensp, gene in zip(anno['string_protein_id'].tolist(), anno['alias'].tolist()):
        anno_dic[ensp].append(gene)


## ============================================================================================
## Network propagation
print('run network propagation, %s'%time.ctime())

for biomarker, feature in zip(ib['Name'].tolist(), ib['Feature'].tolist()):
        if '%s.txt'%biomarker in os.listdir(fo_dir):
                continue
        if not 'target' in feature:
                continue
        print('\ttesting %s, %s'%(biomarker, time.ctime()))

        output = defaultdict(list)
        output_col = ['gene_id', 'string_protein_id', 'propagate_score']

        # network propagation
        biomarker_genes = ib.loc[ib['Name']==biomarker,:]['Gene_list'].tolist()[0].split(':')
        pIDs = annotation.loc[annotation['alias'].isin(biomarker_genes),:]['string_protein_id'].tolist() #only remain biomarker_gene
        propagate_input = {}
        for node in network_nodes:
                propagate_input[node] = 0
                if node in pIDs:
                        propagate_input[node] = 1
        propagate_scores = pagerank(G, personalization=propagate_input, max_iter=100, tol=1e-06) ## NETWORK PROPAGATION

        # output
        for ensp in list(propagate_scores.keys()):
                geneID = 'NA'
                if ensp in list(anno_dic.keys()):
                        for gene in anno_dic[ensp]:
                                geneID = gene
                                output['gene_id'].append(geneID)
                                output['string_protein_id'].append(ensp)
                                output['propagate_score'].append(propagate_scores[ensp])
                else:
                        geneID = 'NA'
                        output['gene_id'].append(geneID)
                        output['string_protein_id'].append(ensp)
                        output['propagate_score'].append(propagate_scores[ensp])
        output = pd.DataFrame(data=output, columns=output_col)
        output.to_csv('%s/%s.txt'%(fo_dir, biomarker), sep='\t', index=False)

end = time.ctime()
print('process complete // start: %s, end: %s'%(start, end))

