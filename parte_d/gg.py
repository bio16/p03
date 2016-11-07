#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import igraph
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import math
#--------------------------------------------------------
def fprobs(g, comms):
    clustering = comms['greedy'].as_clustering()
    membership = clustering.membership
    set_memb = np.array(list(set(membership)))
    # frequentist probability
    fprob = np.zeros(shape=(set_sex.size,set_memb.size), dtype=np.float)
    for i in range(len(g.vs)):
        g.vs[i]["membership"] = membership[i]

    # let's sample both labels && accumulate into `fprob`
    for v in g.vs:
        # catch the membership-index 
        i_m = (v['membership']==set_memb).nonzero()[0][0]
        # catch the sex-index 
        i_s = (v['sex']==set_sex).nonzero()[0][0]
        fprob[i_s,i_m] += 1.0
    # normalize 
    fprob /= fprob.sum()
    # return conj && marginates
    fp = {
    'conj'       : fprob,
    'sex'        : np.sum(fprob, axis=1),
    'membership' : np.sum(fprob, axis=0),
    }
    return fp

# ------- Cargo la informacion del problema ------------- #

# Cargo en el objeto graph la red de dolphins.gml.
graph = igraph.read('../dolphins.gml')

# Leo la info sobre el sexo de los delfines.
dolphins_sex = pd.read_table('../dolphinsGender.txt', header = None)    
_set_sex  = list(set(dolphins_sex))

#--- Le asigno a cada vértice la información sobre el nombre
# y el sexo, reflejado en el color del vertice en el plot.
# Nodo Azul: macho.
# Nodo Rosa: hembra.
# Nodo Verde: sexo desconocido.
color_dict = {'m': "blue", 'f': "pink"}
for i in range(len(dolphins_sex)):
    dolphin_name = dolphins_sex[0][i]
    dolphin_sex = dolphins_sex[1][i]
    for vs in graph.vs:
        if vs['label'] == dolphin_name:
            vs['name'] = dolphin_name
            vs['sex'] = dolphin_sex if dolphin_sex in ('m','f') else 'none'
            #vs['color'] = color_dict[vs['sex']] if vs['sex'] in ('m','f') else 'green'

# catch the three sex types
set_sex  = np.array(list(set([ v['sex'] for v in graph.vs ])))

colours = ['blue','red','green','yellow','orange', 'cyan']
# ----- Fast greedy community detection ----- #
comms = {
'greedy'        : graph.community_fastgreedy(),
'betweenness'   : graph.community_edge_betweenness(directed = False),
'infomap'       : graph.community_infomap(),
'louvain'       : graph.community_multilevel(),
}

fp = fprobs(graph, comms)
#clustering = comms['greedy'].as_clustering()
#membership = clustering.membership
#set_memb = np.array(list(set(membership)))
## frequentist probability
#fprob = np.zeros(shape=(set_sex.size,set_memb.size), dtype=np.float)
#for i in range(len(graph.vs)):
#    graph.vs[i]["membership"] = membership[i]
#    #graph.vs[i]["color"] = colours[membership[i]]
#
## let's sample both labels:
#for v in graph.vs:
#    # catch the membership-index 
#    i_m = (v['membership']==set_memb).nonzero()[0][0]
#    # catch the sex-index 
#    i_s = (v['sex']==set_sex).nonzero()[0][0]
#    fprob[i_s,i_m] += 1.0
## normalize 
#fprob /= fprob.sum()
#
#fp = {
#'conj'       : fprob,
#'sex'        : np.sum(fprob, axis=1),
#'membership' : np.sum(fprob, axis=0),
#}


#EOF
