#!/bin/env python3

import igraph
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib

import random
random.seed(123457)  # fijamos semillas para que todas las comunidades graficadas sean las mismas

graph = igraph.read('dolphins.gml')
plt.style.use(['seaborn-ticks'])

#########################################################################
#  Funciones auxiliares
#########################################################################


def join_probability(partition1,partition2):
    
    matrix = np.zeros(  (max(P[partition1])+1,max(P[partition2])+1)  )

    for key,group in P.groupby([partition1,partition2]):
        i,j = key
        matrix[i,j] = len(group)
    return matrix/np.sum(matrix.flat)

def matrix_plot(partition1,partition2):
    matrix = join_probability(partition1,partition2)
    fig,subplot = plt.subplots(nrows=1,ncols=1,sharey=True,sharex=False)
    cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True)

    cmap = plt.cm.get_cmap('Greys')
    cax = subplot.imshow(matrix, interpolation='none',cmap=cmap)
    subplot.grid(b=False)
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('Probabilidad',fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    yticks = subplot.get_yticks()
    subplot.set_yticklabels( map(lambda x: int(x) if x==int(x) else '',yticks) )
    
    subplot.set_xlabel('comunidad '+p1)
    subplot.set_ylabel('comunidad '+p2)

    for item in ([subplot.title, subplot.xaxis.label, subplot.yaxis.label] +
                         subplot.get_xticklabels() + subplot.get_yticklabels()):
            item.set_fontsize(28)
    plt.tight_layout()
    fig.savefig('informe/figuras/join_proba_'+'-'.join([partition1,partition2])+'.pdf')


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

############################################################
# Informacion Mutua
###########################################################
# Iteramos 1000 veces para calcular la info mutua media

info_mutua = pd.DataFrame({})
partition_list = ['fastgreedy','infomap','louvain','edge_betweeness']
for iter in range(1000):

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




    Proba = {}
    for i,partition in enumerate(P.keys()):
        bins = np.arange(max(P[partition])+2)-.5
       
        proba,bins = np.histogram(P[partition],bins=bins,normed=True)
        Proba[partition] = proba
        if iter == 0:
            fig,subplot = plt.subplots(nrows=1,ncols=1,sharey=True,sharex=False)
            sns.distplot(P[partition],ax=subplot, kde=False, rug=True, norm_hist=True, bins=bins)
            xticks = subplot.get_xticks()

            
            centers = map(lambda x: int(x) if x==int(x) else '',xticks)
            subplot.set_xticklabels(centers)
            subplot.set_xlim(xmin=-0.5)
            for item in ([subplot.title, subplot.xaxis.label, subplot.yaxis.label] +
                                 subplot.get_xticklabels() + subplot.get_yticklabels()):
                    item.set_fontsize(28)
            subplot.set_ylabel('Probabilidad')

            plt.tight_layout()
            plt.savefig('informe/figuras/'+partition+'_probability.pdf')

    P['node'] = range(len(P))


    iter_info = pd.DataFrame({})
    for (p1,p2) in combinations(partition_list,2):
        iter_info[ '-'.join([p1,p2]) ] = [shared_info(p1,p2)]

        if iter == 0:
            matrix_plot(p1,p2)
    info_mutua = pd.concat([info_mutua,iter_info])


print(info_mutua.mean(axis=0))
print(info_mutua.shape)







