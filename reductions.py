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

# Polynomial reductions
import random
from itertools import combinations, count
from planegraphs import Graph, pairwise, set_copy, dict_copy
from pathsearch import shortest_path
from graphio import load_from_edge_list_named, save_to_edge_list_named
from reduce3col import incremental_depth_3COL


#-------------------------------------------------------------------------------
# Polynomial reduction from general 3-colorability to Planar 3-colorability
#-------------------------------------------------------------------------------

class FalsePlanarGraph(Graph):
    false_edges = set()

    def __init__(self):
        Graph.__init__(self)
        self.false_edges = set()

    def add_false_edge(self, u, v):
        self.false_edges.add(frozenset((u, v)))

    def del_false_edge(self, u, v):
        self.false_edges.discard({u, v})

    def copy(self):
        G = self.__class__()

        # copy the minimum set of properties to get an operative graph copy
        G.vertices = set(self.vertices)
        G.edges = set_copy(self.edges)
        G.neighbors = dict_copy(self.neighbors)
        G.first_v = self.first_v
        G.v_id = count(self.v_id.next() - 1)
        self.v_id = count(self.v_id.next() - 2)
        G.identities = dict_copy(self.identities)
        G.false_edges = set_copy(self.false_edges)

        return G

    def planar_copy(self):
        return Graph.copy(self)


    @classmethod
    def from_graph(cls, G):
        H = FalsePlanarGraph()

        map(H.add_named_vertex, G.vertices)
        H.set_vertex_index()

        for u, v in G.edges:
            H.add_edge(u, v)
            if not H.is_planar(set_embedding = False):
                H.remove_edge(u, v)
                H.add_false_edge(u, v)

        return H






def intersect_point(a1, a2, b1, b2):
    da = (a2[0] - a1[0], a2[1] - a1[1])
    db = (b2[0] - b1[0], b2[1] - b1[1])
    dp = (a1[0] - b1[0], a1[1] - b1[1])
    dap = (-da[1], da[0])
    denom = dap[0] * db[0] + dap[1] * db[1]
    num = dap[0] * dp[0] + dap[1] * dp[1]
    num_denom = num / float(denom)
    return (num_denom * db[0] + b1[0], num_denom * db[1] + b1[1])

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
        return intersect_point(A, B, C, D)

    return False

def edges_cross(e1, e2, pos):
    return intersect(pos[e1[0]], pos[e1[1]], pos[e2[0]], pos[e2[1]])

def crossing_to_vertex(G, e1, e2):
    v = G.add_vertex()
    G.add_edge(e1[0], v)
    G.add_edge(e1[1], v)
    G.add_edge(e2[0], v)
    G.add_edge(e2[1], v)
    G.remove_edge(*e1)
    G.remove_edge(*e2)


# #    v = G + 1      # add one vertex to G
# #    G + (e1[0],v)  # add edge
# #    G + (e1[1],v)  # add edge
# #    G + (e2[0],v)  # add edge
# #    G + (e2[1],v)  # add edge
# #    G - (*e1)      # remove edge
# #    G - (*e2)      # remove edge

    return v




def find_crossing(G):
    for e1, e2 in combinations(sorted(G.Edges()), 2):
        if e1 & e2:
            continue
        e1, e2 = sorted((tuple(e1), tuple(e2)))
        ip = edges_cross(e1, e2, G.coordinates)
        if ip:
            return e1, e2, ip

    return False

def set_random_coordinates(G):
    # random positions
    for v in G.Vertices():
        G.coordinates[v] = (random.randint(25, 425), random.randint(25, 425))


def clean_isolated_vertices(G):
    H = G.copy()
    for v in G.Vertices():
        if G.degree(v) == 0:
            H.del_vertex(v)
    return H



def point_distance(p1, p2):
    d = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return d




def planar_gadget(G, x = None, y = None):

    V = [G.add_named_vertex("dummy" + str(G.v_id.next())) for i in range(13)]
    # V = G + 13  # which notation is clearer?


    for u, v in pairwise(V):
        G.add_edge(u, v)

    G.add_edge(V[0], V[7])
    G.add_edge(V[0], V[8])

    G.add_edge(V[1], V[9])
    G.add_edge(V[2], V[9])

    G.add_edge(V[3], V[10])
    G.add_edge(V[4], V[10])

    G.add_edge(V[5], V[11])
    G.add_edge(V[6], V[11])

    for i in (V[8], V[9], V[10]):
        G.add_edge(V[12], i)

    G.add_edge(V[8], V[11])

    return G, V[0], V[2], V[4], V[6]

def uncross_edges(G, e1, edge_path):

    x, y = e1

    DeleteEdge = G.remove_edge
    contract = G.contract
    AddEdge = G.add_edge

    DeleteEdge(*e1)
    for u, v in edge_path:
        DeleteEdge(u, v)
        G, north, east, south, west = planar_gadget(G, x, y)
        contract(x, west)
        contract(u, north)
        AddEdge(south, v)
        x = east

    AddEdge(x, y)

def min_crossing_path(G, u, v):

    D = G.dual()

    if len(G[u]) == 1:
        _, x = {0} | G[u]
        D.vertex_faces[u] = D.vertex_faces[x]
    if len(G[v]) == 1:
        _, x = {0} | G[v]
        D.vertex_faces[v] = D.vertex_faces[v]

    paths = [shortest_path(D, start, D.vertex_faces[v]) for start in D.vertex_faces[u]]

    best = min(paths, key = len)  # path of minimum length
    edge_path = []
    append = edge_path.append

    for f1_id, f2_id in pairwise(best):
        edges = tuple(D.faces_sets[f1_id - 1] & D.faces_sets[f2_id - 1])  # common edges to cross a face boundary
        append(edges[0])




    return edge_path



def reduce_to_planar_3_coloring(G):
    """
    Reduction from 3-coloring to planar 3-coloring
    """

    if G.is_planar():
        return G

    H = FalsePlanarGraph.from_graph(G)

    for e in H.false_edges:
        u, v = e

        # set a combinatorial embedding in H
        H.is_planar()

        # find the minimum crossing path from u to v
        path = min_crossing_path(H, u, v)

        # apply the planar gadget to every crossing
        uncross_edges(H, e, path)

    return H.planar_copy()



#-------------------------------------------------------------------------------
# reduction of 3SAT to 3-colorig
#-------------------------------------------------------------------------------

def reduce_3sat_to_3col(instance, G = None):
    """reduces a 3sat instance to a graph 3-coloring instance
       receives a graph G
        
      for each clause (a,b,c)
      gadget:
      (-a)----(a)---(g1)
                     | \
                     | (g3)---(g4)    (X)             
                     | /       | \   / |
      (-b)----(b)---(g2)       |  (T)  |
                               | /   \ |
      (-c)----(c)-------------(g5)    (F)
      
      X is adjacent to all variables
    """
    G = Graph() if G is None else G

    # add common gadget
    G.add_named_vertex('T')
    G.add_named_vertex('F')
    G.add_named_vertex('X')
    G.add_edge('T', 'F')
    G.add_edge('F', 'X')
    G.add_edge('X', 'T')


    # add gadget for variables
    variables = sorted(set([abs(v) for clause in instance for v in clause]))

    for v in variables:
        G.add_named_vertex(v)
        G.add_named_vertex(-v)
        G.add_edge('X', v)
        G.add_edge('X', -v)
        G.add_edge(v, -v)

    G.set_vertex_index(max(variables) + 1)
    # add the clause gadgets
    for a, b, c in instance:
        g1, g2, g3, g4, g5 = [G.add_vertex() for _i in range(5)]

        # triangle 1,2,3
        G.add_edge(g1, g2)  # 1
        G.add_edge(g2, g3)  # 2
        G.add_edge(g3, g1)  # 3

        # bridge betwen triangle 1,2,3T  and 4,5,T
        G.add_edge(g3, g4)  # 4

        # triangle 3,4,5
        G.add_edge(g4, g5)  # 5
        G.add_edge(g5, 'T')  # 6
        G.add_edge('T', g4)  # 7

        # edges for clause a,b,c
        G.add_edge(a, g1)  # 8
        G.add_edge(b, g2)  # 9
        G.add_edge(c, g5)  # 10

    return G


#----------------------------------------------------------------------------------------
# Reduction from k-coloring to 3-oloring
#----------------------------------------------------------------------------------------

def prepare_grid(G):
    # ensure that k is odd
    if len(G) % 2 == 0:
        G.add_named_vertex('dummykcol')
        for v in G.vertices - {'dummykcol'}: G.add_edge('dummykcol', v)

    H = Graph()
    # Create the color blue
    H.add_named_vertex('b')

    H.add_named_vertex('g')
    H.add_edge('b', 'g')

    return G, H

def create_kgrid(H, N, k):
    # Build a k x n rectangular grid
    for j in range(N):
        i = 0
        u0 = str((i, j))
        H.add_named_vertex(u0)
        H.add_edge('b', u0)

        v0 = str((i, j, 'c'))
        H.add_named_vertex(v0)
        H.add_edge(u0, v0)

        for i in range(1, k):
            u = str((i, j))
            H.add_named_vertex(u)
            H.add_edge('b', u)

            v = str((i, j, 'c'))
            H.add_named_vertex(v)
            H.add_edge(u, v)  # matching mate edge

            H.add_edge(v, v0)  # edges for the k-cycle

            u0 = u
            v0 = v

        v0 = str((0, j, 'c'))
        H.add_edge(v, v0)  # edge for completing the ith k-cycle

    return H

def add_pheripherals_per_edge(edges, H, k):
    for v, w in list(edges):
        for i in range(k):
            # add triangle x y z
            x = str((i, v, w, 'x'))
            y = str((i, v, w, 'y'))
            z = str((i, v, w, 'z'))

            H.add_named_vertex(x)
            H.add_named_vertex(y)
            H.add_named_vertex(z)

            H.add_edge(x, y)
            H.add_edge(y, z)
            H.add_edge(z, x)

            # add three edges
            H.add_edge(x, str((i, v)))
            H.add_edge(y, str((i, w)))
            H.add_edge(z, 'g')

    return H




def reduce_kcol_to_3col(G, k):
    """
    Reduces a k-coloring instance to a 3-coloring one by the Lovasz reduction.
    """

    G, H = prepare_grid(G)
    print("grid prepared")
    N = len(G)
    H = create_kgrid(H, N, k)
    print("grid created")
    H = add_pheripherals_per_edge(G.edges, H, k)
    print("peripherals added")

    return H


def reduction_from_kcol_to_3col(G, k):
    # add the main gadget

    T = lambda *x: "dummy(" + ",".join(map(str, x)) + ')'
    V = lambda v, c: str(v) + '(' + str(c) + ')'
    if k == 3: return G
    H = Graph()
    # Create the color blue
    H.add_named_vertex('c1')
    H.add_named_vertex('c2')
    H.add_named_vertex('c3')
    H.add_edge('c1', 'c2')
    H.add_edge('c1', 'c3')
    H.add_edge('c2', 'c3')

    # add vertices
    for v in G:
        H.add_named_vertex(V(v, 0))
        H.add_edge(V(v, 0), 'c1')
        H.add_edge(V(v, 0), 'c3')

        H.add_named_vertex(V(v, k))
        H.add_edge(V(v, k), 'c2')
        H.add_edge(V(v, k), 'c3')

        for i in range(1, k):
            H.add_named_vertex(V(v, i))
            H.add_edge('c3', V(v, i))

    for u, v in G.edges:
        for i in range(1, k + 1):
            H.add_named_vertex(T(u, v, u, i))
            H.add_edge(V(u, i - 1), T(u, v, u, i))
            H.add_edge(V(u, i), T(u, v, u, i))

            H.add_named_vertex(T(u, v, v, i))
            H.add_edge(V(v, i - 1), T(u, v, v, i))
            H.add_edge(V(v, i), T(u, v, v, i))
            H.add_edge(T(u, v, u, i), T(u, v, v, i))

    H.set_vertex_index()

    return H






#-----------------------------------------------------------------------------------------
def main():
    G = load_from_edge_list_named('kinstances/IF.col')
    # H = reduce_kcol_to_3col(G, k)
    H = reduce_to_planar_3_coloring(G)
    save_to_edge_list_named('kinstances/planar_IF', H, '', 'Reduction from 3-coloring to planar 3-coloring\n')



def main2():
#    from Gato.Gred import DrawMyGraph
#    instance = [[1, 2, 3]]
#    G = reduce_3sat_to_3col(instance)
#    # print list(G.vertices)
#    # print list(G.edges)
#    DrawMyGraph(G)
    from graph_generators import random_connected_graph
    G = random_connected_graph(40, 80)
    k = 100
    # G = load_from_edge_list_named('graphskcolor/dsjc250.5.col')
    # H = reduce_kcol_to_3col(G, k)
    H = reduction_from_kcol_to_3col(G, 4)
    print("reduction done")
    print(len(H))
    print(len(H.edges))
    # for e in H.edges:        print e
    for v in H.vertices:
        print(v)
    # Q, H, P, alpha = incremental_deep_3COL(H, max_alpha = 2)
    # print 'is', k, 'colorable?', bool(Q)




if __name__ == '__main__':
    main()

