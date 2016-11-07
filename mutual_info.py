#!/bin/env python3

import igraph
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations


graph = igraph.read('dolphins.gml')

#dolphins_sex = pd.read_table('dolphinsGender.txt', header = None)


P = pd.DataFrame({})


# Community: fast greedy
community = graph.community_fastgreedy().as_clustering()
P['fastgreedy'] = community.membership

# Community: Infomap
community = graph.community_infomap()
P['infomap'] = community.membership

# Community: Louvain 
community = graph.community_multilevel()
P['louvain'] = community.membership

# Community: Edge_betweeness
community = graph.community_edge_betweenness(directed = False).as_clustering()
P['edge_betweeness'] = community.membership


partition_list = ['fastgreedy','infomap','louvain','edge_betweeness']

fig,subplot = plt.subplots(nrows=1,ncols=4,sharey=True,sharex=False)

Proba = {}
for i,partition in enumerate(P.keys()):
    bins = np.arange(max(P[partition])+2)-.5
    sns.distplot(P[partition],ax=subplot[i], kde=False, rug=True, norm_hist=True, bins=bins)
    xticks = subplot[i].get_xticks()

    centers = map(lambda x: int(x) if x==int(x) else '',xticks)
    subplot[i].set_xticklabels(centers)
   
    proba,bins = np.histogram(P[partition],bins=bins,normed=True)
    Proba[partition] = proba

plt.savefig('label_probability.pdf')

P['node'] = range(len(P))

def join_probability(partition1,partition2):
    
    matrix = np.zeros(  (max(P[partition1])+1,max(P[partition2])+1)  )

    for key,group in P.groupby([partition1,partition2]):
        i,j = key
        matrix[i,j] = len(group)
    return matrix/np.sum(matrix.flat)

def matrix_plot(partition1,partition2):
    matrix = join_probability(partition1,partition2)
    fig,subplot = plt.subplots(nrows=1,ncols=1,sharey=True,sharex=False)
    cax = subplot.imshow(matrix, interpolation='none',cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True))
    subplot.grid(b=False)
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('Probabilidad')
    yticks = subplot.get_yticks()
    subplot.set_yticklabels( map(lambda x: int(x) if x==int(x) else '',yticks) )
    fig.savefig('join_proba_'+'-'.join([partition1,partition2])+'.pdf')


def shared_info(p1,p2,norm=True):
    matrix = join_probability(p1,p2)
    IMAX,JMAX = matrix.shape
    info = 0
    for i in range(IMAX):
        for j in range(JMAX):

            info += matrix[i,j]*np.log( matrix[i,j]/(Proba[p1][i]*Proba[p2][j]) ) if matrix[i,j] > 0 else 0

    if norm:
        Hp1 = 0
        for p in Proba[p1]:
            Hp1 -= p*np.log(p)
        Hp2 = 0
        for p in Proba[p2]:
            Hp2 -= p*np.log(p)

        info = 2*info/(Hp1+Hp2)
    return info


for (p1,p2) in combinations(partition_list,2):
    print(p1,p2,'shared info:',shared_info(p1,p2))






