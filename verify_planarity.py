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
import planegraphs as pg
from Gato.Gred import DrawMyGraph
from planarity_test import FPP_planar_coords, Schnyder_planar_coords, is_planar
from pathsearch import shortest_path
import random
import time
import math



def main():

    times = list()
    vertices = list()
    edges = list()
    N = 50
    M = pg.edges_from_connectivity(N)
    for i in range(100):
        G = pg.random_planar_g(10, 20)

        # D1 = G.dual()

        edges = is_planar(G)
        print "faces"
        for f in G.faces_iterator():
            print f

        print

        G.is_planar()
        print "faces2"
        for f in G.faces_iterator():
            print f


        G.coordinates = FPP_planar_coords(G)
        DrawMyGraph(G, G.coordinates)







# #        G = pg.random_connected_graph(N, M)
# #        #G = pg.load_from_edge_list("test_cross.col")
# #        #G = pg.load_pickled("problem.pic")
# #
# #        #G,x,xp,y,yp = pg.planar_gadget(G)
# #        #print x,xp,y,yp
# #        #G.coordinates = FPP_planar_coords(G)
# #        #DrawMyGraph(G, G.coordinates)
# #
# #        #D = G.dual()
# #        #D.coordinates = FPP_planar_coords(D)
# #
# #        #start, dest = random.sample(D.Vertices(),2)
# #        #print 'from',start,'to',dest
# #        #print shortest_path(D,start,dest)
# #        #print
# #        #DrawMyGraph(D, D.coordinates)
# #
# #        t1 = time.clock()
# #        P = pg.reduce_to_planar_3_coloring(G)
# #        t2 = time.clock()
# #        #if not P: continue
# #
# #        print "size", G.Order(), G.Size(), P.Order(), P.Size(), 'time:', t2 - t1, pg.graph_connectivity(N, M)
# #        times.append(t2 - t1)
# #        vertices.append(P.Order())
# #        edges.append(P.Size())

    # DrawMyGraph(G,pos)
    return times, vertices, edges





if __name__ == '__main__':

    times, vertices, edges = main()
    print 'generated graphs', len(vertices)
    print 'max vertices', max(vertices), 'average', sum(vertices) / (len(vertices) * 1.0), 'max edges', max(edges), 'max time', max(times)



