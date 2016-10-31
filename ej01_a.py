#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Uso la librería igraph y pandas.
import igraph
import pandas as pd

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

colours = ['blue','red','green','yellow','orange']

# ----- Fast greedy community detection ----- #

com = graph.community_fastgreedy()
clustering = com.as_clustering()
for i in range(len(graph.vs)):
    graph.vs[i]["color"] = colours[clustering.membership[i]]

# Grafo Fruchterman - Reingold
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'Fast_greedy.eps')

# ----- Edge betweenness community detection ----- #

com = graph.community_edge_betweenness(directed = False)
clustering = com.as_clustering()
for i in range(len(graph.vs)):
    graph.vs[i]["color"] = colours[clustering.membership[i]]
# Grafo Fruchterman - Reingold
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'Edge_betweenness.eps')

# ----- Infomap community detection ----- #

com = graph.community_infomap()
for i in range(len(graph.vs)):
    graph.vs[i]["color"] = colours[clustering.membership[i]]
# Grafo Fruchterman - Reingold
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'Infomap.eps')

# ----- Louvain community detection ----- #

com = graph.community_multilevel()
for i in range(len(graph.vs)):
    graph.vs[i]["color"] = colours[clustering.membership[i]]
# Grafo Fruchterman - Reingold
layout = graph.layout_fruchterman_reingold()
igraph.plot(graph, layout = layout, target = 'Louvain.eps')
