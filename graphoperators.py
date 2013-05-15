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
from itertools import combinations, count
from collections import defaultdict
import random


# if c++ planarity (Boyer) is available uses it, otherwise use
# pure python (Hopcroft Tarjan) of GATO
try:
    from planartest import is_planar
    print "using c++ planarity library"
except ImportError:
    from planarity_test import is_planar
    print "using pure python planarity_test"



import networkx as nx


def E(G):
    return set(G.edges())

def V(G):
    return set(G.nodes())

def F(G):
    return list(G.faces_iterator())



def random_permutation(iterable, r = None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

class Graph(nx.Graph):
    """
    A python class for graph algorithms.
    operations are INPLACE, this may break python readability
    but speeds up operations.
    vertices, edges are returned as sets.

    >>> G = Graph()    
    >>> u,v,w,x,y,z = G + 6 #add six vertices
    >>> G + (u,v)  # add edge u,v
    >>> (u,v) in G # test wether u,v is an edge
    True
    >>> G - (u,v) # deletes an edge
    >>> (u,v) in G
    False
    >>> G + (u,v)
    >>> G / (u,v) # contract vertices u,v into u and remove v
    >>> G.vertices
    set([1, 3, 4, 5, 6])
    >>> u
    1
    >>> v
    2
    >>> [v for v in G] # iterate trough vertices
    [1, 3, 4, 5, 6]
    >>> G + (u,w)
    >>> G + (u,x)
    >>> G + (u,y)
    >>> G + (u,z)
    >>> G[u]          # neighbors of vertex u
    set([3, 4, 5, 6])
    >>> G - 6         # delete vertex 6
    >>> G[u]
    set([3, 4, 5])

    """

    def __init__(self, first_v = 1):
        self.first_v = first_v  # start vertex ids from this number
        self.v_id = count(self.first_v)
        self.identities = defaultdict(set)  # a dictionary of identified vertices
        self.embedding = None
        self.coordinates = dict()
        self.faces_sets = list()
        self.vertex_faces = dict()
        self.ncontractions = 0
        nx.Graph.__init__(self)




    def __contains__(self, e):
        """Return True if (x,y) is an edge
        (x,y) in G ?
        [x,y] in G ?
        {x,y} in G ?
        """
        if type(e) == type(1): return nx.Graph.__contains__(self, e)
        if len(e) != 2: return nx.Graph.__contains__(self, e)
        return self.has_edge(*e)

    def __getitem__(self, x):
        """Return a set of neighbors of node n,  'G[n]'
        """
        return set(self.neighbors(x))

    def clear(self, first_v = None):

        nx.Graph.clear()
        self.v_id = count(self.first_v) if not first_v else first_v
        self.identities = defaultdict(set)  # a dictionary of identified vertices
        self.embedding = None
        self.faces_sets = set()
        self.vertex_faces = dict()



    def add_vertex(self, dummy = None):
        x = self.v_id.next()
        self.add_node(x)
        self.identities[x].add(x)
        return x

    def add_named_vertex(self, x):
        """ Add an isolated vertex to self
        """
        self.add_node(x)
        self.identities[x].add(x)
        # maximum_number = max(self.vertices(), key = lambda x: x if isinstance(x, int) else 0)
        # if not isinstance(maximum_number, int):  maximum_number = 0
        # self.v_id = count(maximum_number + 1)

    def set_vertex_index(self, i = None):
        """
        set the vertex index for safely inserting new vertices into the graph
        """
        if i is None:
            maximum_number = max(self.vertices(), key = lambda x: x if isinstance(x, int) else 0)
            if not isinstance(maximum_number, int):  maximum_number = 0
            i = maximum_number + 1

        self.v_id = count(i)



    def del_vertex(self, v):
        """ Delete the vertex v and its incident edges
        """
        # TODO: maybe this could be optimized, this is a critical routine.
        # rem_edges = set(map(frozenset,product([v],self.adjLists[v])))
        # self.edges -= rem_edges
        self.remove_node(v)
        del self.identities[v]

    def is_vertex(self, v):
        """ Check whether v is a vertex """
        return self.has_node(v)

    def __add__(self, e):
        """ Add an edge G + (x,y)=e returning nothing
            or
            Add n=e vertices to G, G + n=e returning the vertices added
        """
        if isinstance(e, int):
            if e == 1:
                return self.add_vertex()
            else:
                return [self.add_vertex() for _i in xrange(e)]

        self.add_edge(*e)

    def __sub__(self, v_or_edge):
        """ Deletes edge (x, y) or vertex (x): G-{u,v} or G - v
        """
        if isinstance(v_or_edge, int):
            self.del_vertex(v_or_edge)
        elif len(v_or_edge) == 1:
            self.del_vertex(v_or_edge[0])
        elif len(v_or_edge) == 2:
            self.del_edge(*v_or_edge)

    def is_edge(self, x, y):
        """ Returns 1 if (x,y) is an edge in G"""
        return self.has_edge(x, y)


    def neighbor_set(self, x):
        return set(self.neighbors(x))

    def edge_set(self):
        return set(self.edges())

    def vertex_set(self):
        return set(self.nodes())

    def is_isolated_vertex(self, v):
        """ Returns 1 if the vertex v is isolated"""
        return self.degree(v) == 0

    def is_planar(self, e = None, set_embedding = True):
        """ test if graph is planar
        """
        if e:
            if isinstance(e[0], int): e = [e]
            for x, y in e: self.add_edge(x, y)
            Q = is_planar(self, set_embedding = False)
            for x, y in e: self.del_edge(x, y)
            return bool(Q)

        return bool(is_planar(self, set_embedding))

    def trace_faces(self):
        """
        List the faces of Graph (returned as a list of lists of edges (tuples) of
        the current embedding.

        """
        if not self.embedding:
            self.is_planar()  # get an embedding

        # Establish set of possible edges
        edgeset = self.edges()
        edgeset |= set(map(tuple, map(reversed, edgeset)))

        # Storage for face paths
        path = [edgeset.pop()]
        faces_sets = []


        # Trace faces
        while len(edgeset) > 0:
            neighbors = self.embedding[path[-1][-1]]
            next_node = neighbors[(neighbors.index(path[-1][-2]) + 1) % (len(neighbors))]
            tup = (path[-1][-1], next_node)

            if tup == path[0]:
                faces_sets.append(set([frozenset(i) for i in path]))
                path = [edgeset.pop()]
            else:
                path.append(tup)
                edgeset.discard(tup)

        if len(path):
            faces_sets.append(set([frozenset(i) for i in path]))

        return faces_sets

    def faces_iterator(self):
        """
        iterate over the faces of Graph (returned as a list of lists of edges (tuples) of
        the current embedding.

        """
        if not self.embedding:
            self.is_planar(None, True)  # assure an embedding

        # Establish set of possible edges
        edgeset = set(map(tuple, self.edges()))
        edgeset |= set(map(tuple, map(reversed, edgeset)))

        path = [edgeset.pop()]
        fpath = [path[0][0]]

        while len(edgeset) > 0:

            neighbors = self.embedding[path[-1][-1]]
            next_node = neighbors[(neighbors.index(path[-1][-2]) + 1) % (len(neighbors))]
            tup = (path[-1][-1], next_node)

            if tup == path[0]:
                old_fpath = list(fpath)
                path = [edgeset.pop()]
                fpath = [path[0][0]]
                yield old_fpath
            else:
                path.append(tup)
                fpath.append(tup[0])
                edgeset.discard(tup)

        if len(path):
            yield [e[0] for e in path]

    def dual(self):
        """
        return the dual graph of self
        """

        f = list(self.faces_iterator())
        f_edges = [set(map(frozenset, zip(i, i[1:] + [i[0]])))  for i in f ]

        D = Graph()
        for i in xrange(len(f)):
            D.add_vertex()

        vertex_faces = defaultdict(set)
        AddEdge = D.add_edge

        for i in xrange(len(f)):
            for v in f[i]:
                vertex_faces[v].add(i + 1)

        for i, j in combinations(xrange(len(f)), 2):
            if f_edges[i] & f_edges[j]:
                AddEdge(i + 1, j + 1)

        D.faces_sets = f_edges
        D.vertex_faces = vertex_faces

        return D

    def find_planar_preserving_edge(self):
        for f in self.faces_iterator():
            if len(f) == 3: continue  # not neccesary but for speed up
            u = f[0]
            for v in f[2:-1]:
                if not self.has_edge(u, v):
                    return u, v

        return None

    def get_planar_preserving_edges(self):
        ppe = set()
        for f in random_permutation(self.faces_iterator()):
            if len(f) == 3: continue  # not neccesary but for speed up
            u = f[0]
            for v in f[2:-1]:
                if not self.has_edge(u, v):
                    ppe.add(frozenset((u, v)))

        return ppe

    def is_triangle_free(self):
        """ test if graph is triangle free
        """
        for _t in self.triangles():
            return 0
        return 1

    def planar_density(self):
        """ planar_density w.r.t. to a maximal planar graph (3n-6) """
        return 100.0 * (len(self.edges()) / (3.0 * len(self.vertices()) - 6.0))

    def connectivity(self):
        """
        connectivity w.r.t. to a complete graph V/ (V^2-V)/2
        """
        # return self.order() * self.size() / (self.order() * (self.order() - 1.0) / 2.0)
        return 2.0 * self.size() / float(self.order())

    def avg_degree(self):
        "standard graph theoretic average degree"
        return 2.0 * self.size() / float(self.order())

    def contract(self, x, y):
        """ contract two vertices, e.g. identify it if there is no edge between x,y
        or contract edge x,y if x,y is an edge of G.
        This routine does not insert a new vertex, just copy the neighbors of y into x
        and then deletes vertex y
        This can be optimized by making x the vertex with higher degree
        """
        N = self.neighbor_set
        for z in N(y) - N(x) - {x}:
            self.add_edge(x, z)

        self.identities[x] |= self.identities[y]
        self.del_vertex(y)
        self.ncontractions += 1

    def __div__(self, e):
        self.contract(*e)

    def __truediv__(self, e):
        self.contract(*e)

    def subdivide(self, x, y):
        """
        subdivide edge xy and return new vertex z between x and y
        """
        z = self.add_vertex()
        self.add_edge(x, z)
        self.add_edge(y, z)
        self.del_edge(x, y)
        return z

    def is_complete(self, v_list):
        for x, y in combinations(v_list, 2):
            if {x, y} not in self.edge_set(): return False
        return True

    def is_independent_set(self, v_list):
        for x, y in combinations(v_list, 2):
            if self.has_edge(x, y): return False

        return True

    def is_regular(self, r = None):
        r = self.degree(self.adjLists.keys()[0]) if r is None else r
        for v in self.vertices:
            if self.degree(v) != r: return False

        return True

    def triangles(self):
        N = self.neighbor_set
        for x, y in self.edges_iter():
            for z in N(x) & N(y):
                yield x, y, z

    def four_clique(self):
        N = self.neighbor_set
        for x, y, z in self.triangles():
            for w in N(x) & N(y) & N(z):
                return 1, [x, y, z, w]
        return 0, [None, None, None, None]



def graph_operators(G):
    H = Graph()
    for v in G: H.add_named_vertex(v)
    for u, v in G.edges_iter(): H.add_edge(u, v)
    return H





