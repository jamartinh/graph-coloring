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

import sys
from itertools import combinations

import graphio as gio

def verify_coloring_file(str_graph_file, str_witness_file):
    G = gio.load_from_edge_list(str_graph_file)
    with open(str_witness_file, 'r') as f_col:
        lines = f_col.readlines()

    str_col = ''.join(l for l in lines[8, -1])
    C = eval(str_col)  # reads directly as a formated python str(dict())
    V = set()

    # 1st test: is there any edge in a purported independent set?
    for i in (1, 2, 3):
        for x, y in combinations(C[i], 2):
            if G.is_edge(x, y): return False, 'There is an edge' + str((x, y)) + ' in set ' + str(i) + '\n'
            V |= {x, y}

    # 2nd test: are all edges *** OF GRAPH G! *** included in C?
    if sorted(G.Vertices()) != sorted(V): return False, 'Not all vertices of G are included in the purported solution \n'


    return True


def verify_coloring_dict(str_graph_file, W):
    G = gio.load_from_edge_list(str_graph_file)
    with open(str_witness_file, 'r') as f_col:
        lines = f_col.readlines()

    str_col = ''.join(l for l in lines[8, -1])
    C = eval(str_col)  # reads directly as a formated python str(dict())
    V = set()

    # 1st test: is there any edge in a purported independent set?
    for i in (1, 2, 3):
        for x, y in combinations(C[i], 2):
            if G.is_edge(x, y): return False, 'There is an edge' + str((x, y)) + ' in set ' + str(i) + '\n'
            V |= {x, y}

    # 2nd test: are all edges *** OF GRAPH G! *** included in C?
    if sorted(G.Vertices()) != sorted(V): return False, 'Not all vertices of G are included in the purported solution \n'


    return True



if __name__ == '__main__':
    graph_file = sys.argv[1]
    witness_file = sys.argv[2]
    Q, str_text = verify_coloring(graph_file, witness_file)
    print Q, str_text
