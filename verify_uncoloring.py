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

__doc__ = """
A '3-uncoloring' (a 3-uncolorability witness) consist of a sequence of vertex
identification (vertex merging) pairs (x,y) of G such that:

1)  x,y belongs to a $K_{112}$ subgraph of G where x,y is an independent  set
of the $K_{112}$ subgraph.

or

2) x,y belongs to a $C_4$ (square) or $T_{31}$ ((3,1)-tadpole) subgraph T of H where x,y is an independent
set of T and adding the edge x,y to G leads to a sequence of vertex identifications following condition 1)

Then the grahp collapses to a graph containing $K_4$ as a subgraph.
"""

import sys


import graphio as gio

def verify_word(G, word):


    if word[0] == 'K112':
        # verify that is really a K112
        _kind, zw, xw, yw, xyzw, _l = word
        zw = eval(zw)
        xw = eval(xw)
        yw = eval(yw)
        xyzw = eval(xyzw)

        w = set(zw) & set(xw) & set(yw)
        xyz = set(zw) ^ set(xw) ^ set(yw)
        if not G.is_complete(xyz): return False, str(xyz) + ' is not a triangle'  # assure xyz is a triangle
        if not G.is_edge(*xw) or not G.is_edge(*yw): return False, str(xw) + ' or ' + str(yw) + ' is not an edge'  # assure w is joined to x and y
        if w in xyz : return False, str(xyzw) + ' is not a K112'  # assure w is different from z
        G.contract(*zw)


    elif word[0] == 'T31':
        # we will not check nothing since an error here will imply and error
        wy = eval(word[1])
        G.contract(*wy)
        # return prefix+'T31  '+str(O[1])+' '+str(O[2])+'\n'+'BEGIN\n'+UNCOL_witness(O[3],'SUBT31')+'\nEND' + '\n'

    elif word[0] == 'Q.E.D.':
        if word[1] == 'K3 Free':
        # verify that there is no triangles
            for T in G.triangles():  return False, 'K3 Free but triangle ' + str(T) + ' found'
        elif word[0] == 'K4':
            K4 = word[0]
            if len(K4) < 4 or not G.is_complete(K4): return False, str(K4) + ' is not a K4'  # assure K4 is complete


    elif word[0] == 'NOTP':
        if G.is_planar():
            return False, 'NOTP found, but the graph is planar'

    elif word[0] == 'MAXP':
        return False, 'This witness is invalid, some error ocurred or a conunter example to the algorithm was found'


def verify_uncoloring(str_graph_file, str_witness_file):
    G = gio.load_from_edge_list(str_graph_file)
    with open(str_witness_file, 'r') as f_uncol:
        lines = f_uncol.readlines()


    # verify line by line up to Q.E.D. is found
    for line in lines[8:-1]:
        word = line.split()
        if not verify_word(G, word): return False





if __name__ == '__main__':
    str_graph_file = sys.argv[1]
    str_witness_file = sys.argv[2]
    Q, text = verify_uncoloring(str_graph_file, str_witness_file)
    print Q, text
