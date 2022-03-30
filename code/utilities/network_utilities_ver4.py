import networkx as nx
from collections import defaultdict, OrderedDict
import random, os, time, scipy, json
import numpy as np
import scipy.stats as stat
import pandas as pd
import numpy, scipy, pandas, networkx
import csv

        

# RETURN LARGEST CONNECTED COMPONENT (LCC)
def return_LCC( G, geneList ):
	"""
	INPUT
	G : network
	geneList : list of genes
	------------------------------------------------------------------------------------------------
	OUTPUT
	Returns a list of genes from geneList that form a largest connected component (LCC) in network G
	Returns node-edge information as dictionary for LCC genes
	Returns GC (Giant Component)
	"""
	edgeDic = defaultdict(list)
	g = nx.Graph()
	if len(geneList) == 0:
		GC = []
	
	else:
		for i, g1 in enumerate(geneList):
			for j, g2 in enumerate(geneList):
				if i<j:
					if G.has_edge(g1,g2) == True:
						g.add_edge(g1,g2)
		if len(g) > 0:
			GC = max(nx.connected_component_subgraphs(g), key=len)
			for key in GC:
				edgeDic[key] = GC[key].keys()
		else:
			GC = g
	return edgeDic.keys(), edgeDic, GC

