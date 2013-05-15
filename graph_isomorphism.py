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
from collections import defaultdict
from graphoperators import graph_operators, V
from itertools import combinations, izip
import operator
# from planegraphs import V

def isomorphisms(G, H):

    isoset = defaultdict(set)
    # simple check degrees
    degG = sorted((G.degree(v) for v in V(G)))
    degH = sorted((H.degree(v) for v in V(H)))
    print degG
    print degH
    if any((x != y for x, y in izip(degG, degH))): return False, (degG, degH)

    remset = V(H)
    for u in V(G):
        isoset[u] = V(H)
        for v in V(G) - {u}:
            for x, y in combinations(V(H), 2):
                if ((u, v) in G) != ((x, y) in H): continue
                if len(G[u] & G[v]) != len(H[x] & H[y]): continue
                if len(G[u] ^ G[v]) != len(H[x] ^ H[y]): continue
                isoset[u, v] |= {x, y}
                remset -= {x, y}

            isoset[u] &= isoset[u, v]
            del isoset[u, v]
            if len(isoset[u]) == 0: return False, ((u, v), (x, y))
    if len(remset) > 1: return False, remset
    # Borrar el 1, borrar el 2, borrar el 3, etc etc, si queda vacio antes de llegar al ultimo, no hay solucion, i.e. no hay independent set
    # Es un recorrido paralelo por el espacio de soluciones, la interseccion de caminos algoritmicos!
    # Si se eliminan mas clausulas que variables eliminadas entonces no hay solucion.
    # Al borrar variables algunas clausulas deben tener un valor.


    return True, isoset



def test_isoset():
    import networkx as nx
    import matplotlib.pyplot as plt
    G = graph_operators(nx.random_regular_graph(5, 10))
    H = graph_operators(nx.random_regular_graph(5, 10))



    Q, isoset = isomorphisms(G, H)


    if Q:
        isoset = sorted(isoset.iteritems(), key = operator.itemgetter(1))
        for v in isoset:
            print(v[0], sorted(v[1]))
    else:
        print "non isomorphic", sorted(isoset)

    print "networkx:", nx.is_isomorphic(nx.Graph(G), nx.Graph(H))
#    plt.figure(1)
#    nx.draw_spring(G)
#    plt.figure(2)
#    nx.draw_spring(H)
#    plt.show()



if __name__ == "__main__":
    test_isoset()
