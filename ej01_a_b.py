#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Uso la librería igraph y pandas.
import igraph
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from random import shuffle

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

# ----- Fast greedy community detection ----- #
com = graph.community_fastgreedy()
clustering = com.as_clustering()
membership = clustering.membership
for i in range(len(graph.vs)):
    graph.vs[i]["membership"] = membership[i]
    graph.vs[i]["color"] = colours[membership[i]]

# Grafo
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'Fast_greedy.eps')

# Modularidad
print 'Fast greedy modularity:', graph.modularity(membership)

# Modularidad en red recableada
plt.figure(1)
plt.axes([0.10, 0.20, 0.80, 0.70])
random_modularity = []
for i in range(1000):
    graph_aux.rewire(1000)
    random_modularity.append(graph_aux.modularity(membership))
plt.hist(random_modularity, normed = True)

# ----- Edge betweenness community detection ----- #

com = graph.community_edge_betweenness(directed = False)
clustering = com.as_clustering()
membership = clustering.membership
for i in range(len(graph.vs)):
    graph.vs[i]["membership"] = membership[i]
    graph.vs[i]["color"] = colours[membership[i]]

# Grafo
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'Edge_betweenness.eps')

# Modularidad
print 'Edge betweenness modularity: ', graph.modularity(membership)

random_modularity = []
for i in range(1000):
    graph_aux.rewire(1000)
    random_modularity.append(graph_aux.modularity(membership))
plt.hist(random_modularity, normed = True)


# ----- Infomap community detection ----- #

com = graph.community_infomap()
membership = com.membership
for i in range(len(graph.vs)):
    graph.vs[i]["membership"] = membership[i]
    graph.vs[i]["color"] = colours[membership[i]]

# Grafo
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'Infomap.eps')

# Modularidad
print 'Infomap modularity: ', graph.modularity(membership)

random_modularity = []
for i in range(1000): 
    graph_aux.rewire(1000)
    random_modularity.append(graph_aux.modularity(membership))
plt.hist(random_modularity, normed = True)

# ----- Louvain community detection ----- #

com = graph.community_multilevel()
membership = com.membership
for i in range(len(graph.vs)):
    graph.vs[i]["membership"] = membership[i]
    graph.vs[i]["color"] = colours[membership[i]]

# Grafo
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'Louvain.eps')

# Modularidad
print 'Louvain: ', graph.modularity(membership)

#plt.figure(4)
random_modularity = []
for i in range(1000):
    graph_aux.rewire(1000)
    random_modularity.append(graph_aux.modularity(membership))

plt.hist(random_modularity, normed = True)

plt.xlabel('Modularidad', fontsize = 20)
plt.title('Red recableada', fontsize = 20)
plt.grid('on')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('Modularidad_random.eps')
plt.show()

