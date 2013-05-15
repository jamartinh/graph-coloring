# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:14:24 2013

@author: jamh
"""
import pyximport
pyximport.install(pyimport = True)

from reduceKclique import *
import planegraphs
import graphio as gio
import testutil as tu
#-------------------------------------------------------------------------------------------
# TESTING
#-------------------------------------------------------------------------------------------
def main(strFileName, alpha = 5):
    
    print " *** Starting *** "
    G = gio.load_from_edge_list_named(strFileName)
    b, H = greedy_clique(G.copy())
    print b, "clique by greedy"

    V, E, avgdeg = G.order(), G.size(), G.avg_degree()
    t1 = time.clock()
    # k, G, P, last_good = max_clique(G, alpha)
    Q = True
    k = 33
    # x, y = G.find_clique(k)
    # print "find_clique", sorted(x)
    # Q, P = True, None
    # Q, G, P = kclique(G, k , alpha = 20)
    Q, G, P = has_kclique(G, k , alpha = 2)
    t2 = time.clock()
    k = len(G)
    strScreen = "Q: %s k: %d N: %d M: %d time1: %3.3f average degree: %2.2f " % (Q, k, V, E, t2 - t1, avgdeg)
    print strScreen
    # print G.vertices
    # print len(G.vertices)
    witness = YES_witness(G, k, P)
    # if not Q:  witness = r3.UNCOL_witness(P)
    # else: witness = r3.COL_witness(G, P)

    tu.save_witness("", k, witness, "", strFileName)


if __name__ == "Main":
    main(sys.argv[1], sys.argv[2])


# main("to_3coloring_uf50-0955.cnf.col", 5)
main("C125.9.clq", 2)
# main("DSJC500_5.clq", 2)
