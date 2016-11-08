#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import igraph
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import math
from pylab import figure, close
import matplotlib.cm as cm
import funcs as ff
#--------------------------------------------------------

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

# catch the three sex types
#set_sex  = np.array(list(set([ v['sex'] for v in graph.vs ])))
set_sex  = np.array(['m','none', 'f'])

colours = ['blue','red','green','yellow','orange', 'cyan']
# ----- Fast greedy community detection ----- #
comms = {
'greedy'        : graph.community_fastgreedy(),
'betweenness'   : graph.community_edge_betweenness(directed = False),
'infomap'       : graph.community_infomap(),
'louvain'       : graph.community_multilevel(),
}


for aname in comms.keys():
    fp = ff.fprobs(graph, set_sex, comms, aname=aname)
    I  = ff.information(fp['conj'], fp['sex'], fp['membership'])
    print(aname, ': ', I)
    #--- fig
    fig = figure(1, figsize=(5,3))
    ax  = fig.add_subplot(111)
    #ax.contourf(fp['conj'],)
    ax.imshow(fp['conj'], cmap=cm.gray_r, interpolation='none')
    # black: high values
    # white: low values
    ax.set_ylim(-0.49,2.49)
    fig.savefig('p12_'+aname+'.png', dpi=100, bbox_inches='tight')
    close(fig)

print(set_sex)
#EOF
