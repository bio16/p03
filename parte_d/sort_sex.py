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

for i in range(len(dolphins_sex)):
    dolphin_name = dolphins_sex[0][i]
    dolphin_sex = dolphins_sex[1][i]
    for vs in graph.vs:
        if vs['label'] == dolphin_name:
            vs['name'] = dolphin_name
            vs['sex'] = dolphin_sex if dolphin_sex in ('m','f') else 'none'

# Cargo en el diccionario `nsex`, la cantidad de ejemplares
# macho, hembra, e indefinidos.
nsex = {}
nsex['f']  = 0
nsex['m']  = 0
nsex['none'] = 0
for v in graph.vs:
    nsex[v['sex']] += 1

# Creo una lista con todas las etiquetas de género, respetando
# las cantidades de la red original.
list_genders = ['m'] * nsex['m'] + ['f']*nsex['f'] + \
                  ['none'] * nsex['none']


n = ff.count_intergender_links(graph, list_genders)

#EOF
