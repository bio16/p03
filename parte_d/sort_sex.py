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
import random
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import random
random.seed(123457)
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

# nro de realizaciones para los sorteos de sexo
nconf = 15000
# nro de links entre generos (para red original)
ninter_orig = ff.count_intergender_links(graph, list_genders)
# coleccion (entre diferentes realizaciones del sorteo de genero) del 
# nro de link entre generos
ninter = [] 
# seed to reproduce this same set of
# realizations
random.seed(448) 
for conf in range(nconf):
    random.shuffle(list_genders)
    ninter += [ ff.count_intergender_links(graph, list_genders) ]

#--- mean && sigma
avr, sig = np.mean(ninter), np.std(ninter)

#--- histogram of realizations
fig = figure(1, figsize=(6,4))
ax  = fig.add_subplot(111)

ax.hist(ninter, bins=25, normed=True, alpha=0.7, label='$N=%d$'%nconf)
ax.axvline(x=ninter_orig, lw=3, c='black', label='original')
#ax.axvline(x=avr-sig, lw=3, c='blue', label='$\mu - \sigma$')

#--- sombra para sigma
trans   = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
rect1   = patches.Rectangle((avr-sig, 0.0), width=2.*sig, height=1,
            transform=trans, color='blue',
            alpha=0.25, label='$(\mu-\sigma:\mu+\sigma)$')
ax.add_patch(rect1)

#--- otros
ax.set_xlabel('nro de enlaces entre generos')
ax.set_ylabel('densidad de probabilidad')
ax.grid(True)
ax.legend(loc='best')
fig.savefig('./hist_sort_sex.png', dpi=135, bbox_inches='tight')
close(fig)


#EOF
