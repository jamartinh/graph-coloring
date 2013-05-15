#-------------------------------------------------------------------------------
# Copyright (c) 2013 Jose Antonio Martin H. (jamartinh@fdi.ucm.es).
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Public License v3.0
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/gpl.html
#
# Contributors:
#     Jose Antonio Martin H. (jamartinh@fdi.ucm.es) - initial API and implementation
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import cPickle as pickle
from planegraphs import Graph, V
#-------------------------------------------------------------------------------
# File load/save
#-------------------------------------------------------------------------------


def nvertex(v):
    try:
        return int(v)
    except ValueError:
        return v


def load_from_edge_list(strFileName, G = None):
    if G is None: G = Graph()

    with open(strFileName, 'r') as f:
        lines = f.readlines()

    if not G: G = Graph()
    v_list = dict()

    for line in lines:
        if len(line) == 0 or line == '\n': continue
        if line == "": continue
        fields = line.split()
        if not len(fields): continue
        if fields[0] == 'e':
            u = int(fields[1])
            v = int(fields[2])

            if u not in v_list: v_list[u] = G.add_vertex()
            if v not in v_list: v_list[v] = G.add_vertex()

            x, y = v_list[u], v_list[v]
            G.add_edge(x, y)



    return G

def load_from_edge_list_named(strFileName, G = None):
    with open(strFileName, 'r') as f:
        lines = f.readlines()

    if G is None: G = Graph()

    for line in lines:
        if len(line) == 0 or line == '\n': continue
        fields = line.split()
        if not len(fields): continue
        if fields[0] == 'e':
            u, v = nvertex(fields[1]), nvertex(fields[2])
            if u not in V(G):
                G.add_named_vertex(u)
            if v not in V(G):
                G.add_named_vertex(v)
            G.add_edge(u, v)


    G.set_vertex_index()
    return G

def save_to_edge_list(strFile, G, path, prologue = ""):
    N = G.order()
    M = G.size()
    f = open(path + strFile + ".col", "w")
    f.write(prologue)
    str_text = ""
    str_text += "p edge " + str(N) + " " + str(M) + "\n"
    v_list = sorted(G.vertices)
    for x, y in sorted(G.edges()):
        str_text += "e " + str(v_list.index(x) + 1) + " " + str(v_list.index(y) + 1) + "\n"

    f.write(str_text)
    f.close()

def save_to_edge_list_named(strFile, G, path = "", prologue = ""):
    N = G.order()
    M = G.size()
    f = open(path + strFile + ".col", "w")
    f.write(prologue)
    str_text = ""
    str_text += "p edge " + str(N) + " " + str(M) + "\n"
    for x, y in G.edges:
        str_text += "e " + str(x).replace(' ', '') + " " + str(y).replace(' ', '') + "\n"

    f.write(str_text)
    f.close()

def save_pickled(strFile, G, path):
    pickle.dump(G, open(path + strFile, "wb"))

def load_pickled(strFile):
    return pickle.load(open(strFile))

