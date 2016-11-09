#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Uso la librería igraph y pandas.
import igraph
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import random
from random import shuffle
import numpy as np
import math

# ------- Cargo la informacion del problema ------------- #

# Cargo en el objeto graph la red de dolphins.gml.
graph = igraph.read('dolphins.gml')

# Leo la info sobre el sexo de los delfines.
dolphins_sex = pd.read_table('dolphinsGender.txt', header = None)    

# Le asigno a cada vértice la información sobre el nombre
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
            vs['sex'] = dolphin_sex
            try:
                vs["color"] = color_dict[vs['sex']]
            except:
                vs["color"] = "green"
                vs["sex"] = None
colours = ['blue','red','green','yellow','orange', 'cyan']

graph_aux = deepcopy(graph)


# Definicion de silhouette
def silhouette(graph_aux2, membership):
    silhouette_per_vertex = []
    membership_set = set(membership)

    for i in range(len(graph_aux2.vs)):

       average_distance = {}

       for j in range(len(graph_aux2.vs)):

           try:
               average_distance[graph_aux2.vs[j]['membership']].append(graph_aux2.shortest_paths_dijkstra(i,j)[0][0])
           except:
               average_distance[graph_aux2.vs[j]['membership']] = []
               average_distance[graph_aux2.vs[j]['membership']].append(graph_aux2.shortest_paths_dijkstra(i,j)[0][0])

       a = np.mean(average_distance[graph_aux2.vs[i]['membership']])
       
       b_list = [np.mean(average_distance[key]) for key in membership_set if key != graph_aux2.vs[i]['membership']]
       b = min(b_list)

       if (b - a) != 0.00:
           silhouette_per_vertex.append(float(b - a)/max(a,b))
       else:
           silhouette_per_vertex.append(0.00)
        
    return np.mean(silhouette_per_vertex)


# ----- Fast greedy community detection ----- #
random.seed(123457)
com = graph.community_fastgreedy()
clustering = com.as_clustering()
membership = clustering.membership
for i in range(len(graph.vs)):
    graph.vs[i]["membership"] = membership[i]
    graph.vs[i]["color"] = colours[membership[i]]

# Grafo
random.seed(123457)
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'graph_Fast_greedy.png')

# Modularidad y Silhouette
print('Fast greedy modularity:', graph.modularity(membership))
print('Silhouette:', silhouette(graph, membership))

plt.figure(1)
plt.axes([0.10, 0.20, 0.80, 0.70])
plt.figure(2)
plt.axes([0.10, 0.20, 0.80, 0.70])

# Modularidad y solhouette en red recableada
random.seed(123457)
graph_aux = deepcopy(graph)
random_modularity = []
random_silhouette = []
for i in range(1000):
    graph_aux.rewire(1000)
    random_modularity.append(graph_aux.modularity(membership))
    sil = silhouette(graph_aux, membership)
    if math.isnan(sil) == False:
        random_silhouette.append(sil)

plt.figure(1)
plt.hist(random_modularity, normed = True, alpha=0.6)

plt.figure(2)
plt.hist(random_silhouette, normed = True, alpha=0.6)

# ----- Edge betweenness community detection ----- #

random.seed(123457)
com = graph.community_edge_betweenness(directed = False)
clustering = com.as_clustering()
membership = clustering.membership
for i in range(len(graph.vs)):
    graph.vs[i]["membership"] = membership[i]
    graph.vs[i]["color"] = colours[membership[i]]

# Grafo
random.seed(123457)
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'graph_Edge_betweenness.png')

# Modularidad
print('Edge betweenness modularity: ', graph.modularity(membership))
print('Silhouette:', silhouette(graph, set(membership)))


# Modularidad y solhouette en red recableada
random.seed(123457)
graph_aux = deepcopy(graph)
random_modularity = []
random_silhouette = []
for i in range(1000):
    graph_aux.rewire(1000)
    random_modularity.append(graph_aux.modularity(membership))
    sil = silhouette(graph_aux, membership)
    if math.isnan(sil) == False:
        random_silhouette.append(sil)

plt.figure(1)
plt.hist(random_modularity, normed = True, alpha=0.6)

plt.figure(2)
plt.hist(random_silhouette, normed = True, alpha=0.6)

# ----- Infomap community detection ----- #

random.seed(123457)
com = graph.community_infomap()
membership = com.membership
for i in range(len(graph.vs)):
    graph.vs[i]["membership"] = membership[i]
    graph.vs[i]["color"] = colours[membership[i]]

# Grafo
random.seed(123457)
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'graph_Infomap.png')

# Modularidad
print('Infomap modularity: ', graph.modularity(membership))
print('Silhouette:', silhouette(graph, set(membership)))

# Modularidad y solhouette en red recableada
random.seed(123457)
graph_aux = deepcopy(graph)
random_modularity = []
random_silhouette = []
for i in range(1000):
    graph_aux.rewire(1000)
    random_modularity.append(graph_aux.modularity(membership))
    sil = silhouette(graph_aux, membership)
    if math.isnan(sil) == False:
        random_silhouette.append(sil)

plt.figure(1)
plt.hist(random_modularity, normed = True, alpha=0.6)

plt.figure(2)
plt.hist(random_silhouette, normed = True, alpha=0.6)


# ----- Louvain community detection ----- #

random.seed(123457)
com = graph.community_multilevel()
membership = com.membership
for i in range(len(graph.vs)):
    graph.vs[i]["membership"] = membership[i]
    graph.vs[i]["color"] = colours[membership[i]]

# Grafo
random.seed(123457)
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'graph_Louvain.png')

# Modularidad
print('Louvain: ', graph.modularity(membership))
print('Silhouette:', silhouette(graph, set(membership)))

# Modularidad y solhouette en red recableada
random.seed(123457)
graph_aux = deepcopy(graph)
random_modularity = []
random_silhouette = []
for i in range(1000):
    graph_aux.rewire(1000)
    random_modularity.append(graph_aux.modularity(membership))
    sil = silhouette(graph_aux, membership)
    if math.isnan(sil) == False:
        random_silhouette.append(sil)

plt.figure(1)
plt.hist(random_modularity, normed = True, alpha=0.6)

plt.figure(2)
plt.hist(random_silhouette, normed = True, alpha=0.6)

plt.figure(1)
plt.xlabel('Modularidad', fontsize = 20)
plt.title('Red recableada', fontsize = 20)
plt.grid('on')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('Modularidad_random.png')

plt.figure(2)
plt.xlabel('Silhouette', fontsize = 20)
plt.title('Red recableada', fontsize = 20)
plt.grid('on')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('Silhouette_random.png')


plt.show()

