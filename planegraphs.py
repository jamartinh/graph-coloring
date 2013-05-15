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
from collections import defaultdict, deque
from copy import deepcopy
from itertools import combinations, izip, tee, count
import itertools
import random
# if c++ planarity (Boyer) is available uses it, otherwise use
# pure python (Hopcroft Tarjan) of GATO
try:
    from planartest import is_planar
    # import raise_error
    # from nxtools import planarity
    print "using c++ planarity library"
    # is_planar = lambda G, dummy: planarity.is_planar(G.neighbors)
except ImportError:
    from planarity_test import is_planar
    print "using pure python planarity_test"




#-------------------------------------------------------------------------------
# programming routines
#-------------------------------------------------------------------------------

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def random_permutation(iterable, r = None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def probabilistic_choice(iterable, probabilities):
    "random selection from a probability planar_density function"
    rnd_test = random.random()
    p_sum = 0.0
    for i, p in izip(iterable, probabilities):
        if rnd_test <= p + p_sum:
            return i
        p_sum += p
    return None


################################################################################
#  Much faster methods than deepcopy
################################################################################

def dict_copy(source):
    """ copy dict of sets"""
    # return dict((k, set(v)) for (k, v) in source.iteritems())
    # return dict(zip(source.iterkeys(), map(set, source.itervalues())))
    return {k:set(v) for k, v in source.iteritems() }
    # return {k:v for k, v in zip(source.keys(), map(set, source.values()))}

def list_copy(source):
    """ copy list of iterable to list of lists"""
    return map(list, source)

def set_copy(source):
    """copy set of sets"""
    return set(map(frozenset, source))



################################################################################
#
# Graph data structure
#
################################################################################
def E(G):
    return G.edges

def V(G):
    return G.vertices

def F(G):
    return list(G.faces_iterator())

def N(x, G):
    return G.neighborhood(x)

class Graph:
    """
    A python class for graphs
    most operations are INPLACE, this may break readability
    so be carefull

    >>> G = Graph()
    >>> u,v,w,x,y,z = [G.add_vertex() for i in range(6)] #add six vertices
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
        """
        Constructor method.
        """

        self.vertices = set()
        self.edges = set()
        self.neighbors = defaultdict(set)
        self.degrees = defaultdict(int)
        self.faces_sets = list()
        self.vertex_faces = dict()
        self.first_v = first_v  # start vertex ids from this number
        self.v_id = count(self.first_v)
        self.identities = defaultdict(set)  # a dictionary of identified vertices
        self.embedding = None
        self.coordinates = dict()
        self.ncontractions = 0

    def __iter__(self):
        """Iterate over the vertices, i.e. for x in G
        """
        return iter(self.vertices)

    def __contains__(self, e):
        """Return True if (x,y) is an edge
        (x,y) in G ?
        [x,y] in G ?
        {x,y} in G ?
        """
        return set(e) in self.edges

    def __len__(self):
        """
        Returns the number of vertices in G.
        """

        return len(self.vertices)

    def __getitem__(self, x):
        """Return a set of neighbors of node n,  'G[n]'
        """
        return self.neighbors[x]

    def clear(self, first_v = None):
        """
        Clears all the graph data.
        """

        self.vertices = set()
        self.edges = set()
        self.neighbors = defaultdict(set)
        self.degrees = defaultdict(int)
        self.v_id = count(self.first_v) if not first_v else first_v
        self.identities = defaultdict(set)  # a dictionary of identified vertices
        self.embedding = None
        self.faces_sets = list()
        self.vertex_faces = dict()



    def copy(self):
        """
        Return a copy of current graph, faster than deepcopy()
        """

        G = self.__class__()

        # copy the minimum set of properties to get an operative graph copy
        G.vertices = set(self.vertices)
        G.edges = set(self.edges)
        # G.neighbors = defaultdict(set)
        G.neighbors = defaultdict(set, dict_copy(self.neighbors))
        G.degrees = self.degrees.copy()
        G.first_v = self.first_v
        G.v_id = count(self.v_id.next() - 1)
        self.v_id = count(self.v_id.next() - 2)  # restore the current number in self.
        G.identities = defaultdict(set, dict_copy(self.identities))
        # G.identities = dict_copy(self.identities)

        # if not enough, you can always use deepcopy after all!

        return G

    def subgraph(self, vertex_subset, edge_subset = None):
        """Return the subgraph induced on nodes in vertex_subset.

        The induced subgraph of the graph contains the nodes in vertex_subset
        and the edges between those nodes.

        Parameters
        ----------
        vertex_subset : list, iterable
            A container of nodes which will be iterated through once.

        Returns
        -------
        G : Graph
            A subgraph of the graph with the same edge attributes.

        """

        # create new graph and copy subgraph into it
        G = self.__class__()
        # copy the minimum set of properties to get an operative graph copy
        G.vertices = set(vertex_subset)
        superset = G.vertices.issuperset
        if edge_subset:
            G.edges = set(filter(superset, edge_subset))
            for u, v in edge_subset:
                G.neighbors[u].discard(v)
                G.neighbors[v].discard(u)
        else:
            G.edges = set(filter(superset, self.edges))

        G.neighbors.update({v: self.neighbors[v] & G.vertices for v in G.vertices})
        G.degrees.update({v:len(self.neighbors[v]) for v in G.vertices})
        G.identities = defaultdict(set, dict_copy(self.identities))
        G.set_vertex_index()

        return G

    def __call__(self, vertex_subset):
        return self.subgraph(vertex_subset)

    def add_vertex(self, dummy = None):
        """ add a new vertex to G, assiging an integer name to it
        """
        x = self.v_id.next()
        self.vertices.add(x)
        self.identities[x].add(x)
        self.neighbors[x].clear()
        return x

    def add_named_vertex(self, x):
        """ Add an isolated vertex to self
        """

        self.vertices.add(x)
        self.identities[x].add(x)
        self.neighbors[x].clear()
        # maximum_number = max(self.vertices, key = lambda x: x if isinstance(x, int) else 0)
        # if not isinstance(maximum_number, int):  maximum_number = 0
        # self.v_id = count(maximum_number + 1)
        return x

    def set_vertex_index(self, i = None):
        """
        set the vertex index for safely inserting new vertices into the graph
        """

        if i is None:
            if len(self.vertices) == 0:
                self.v_id = count(1)
                return
            maximum_number = max(self.vertices, key = lambda x: x if isinstance(x, int) else 0)
            if not isinstance(maximum_number, int):  maximum_number = 0
            i = maximum_number + 1

        self.v_id = count(i)



    def remove_vertex(self, v):
        """ Delete the vertex v and its incident edges
        """
        # TODO: maybe this could be optimized, this is a critical routine.
        # rem_edges = set(map(frozenset,product([v],self.neighbors[v])))
        # self.edges -= rem_edges

#         self.neighbors.update({u: self.neighbors[u] - {v} for u in self.neighbors[v]})
#         self.degrees.update({u: self.degrees[u] - 1 for u in self.neighbors[v]})
#         self.edges = {e for e in self.edges if v not in e}


        discard = self.edges.discard
        for u in self.neighbors[v]:
            self.neighbors[u].discard(v)
            self.degrees[u] -= 1
            discard({u, v})

        del self.neighbors[v]
        self.vertices.discard(v)
        del self.identities[v]
        del self.degrees[v]

    def has_vertex(self, v):
        """ Check whether v is a vertex """
        return v in self.vertices

    def add_edge(self, u, v):
        """ Add an edge (u,v). Returns nothing
        """
        if u == v: raise KeyError
        self.neighbors[u].add(v)
        self.neighbors[v].add(u)
        self.edges.add(frozenset((u, v)))
        self.degrees[u] += 1
        self.degrees[v] += 1

    def __add__(self, e):
        """ Add an edge G + (x,y)=e returning nothing
            or
            Add n=e vertices to G, G + n, returning the vertices added
        """
        if isinstance(e, int):
            if e == 1:
                return self.add_vertex()
            else:
                return [self.add_vertex() for _i in xrange(e)]

        self.add_edge(*e)


    def remove_edge(self, u, v):
        """ Deletes edge (tail,head).  """
        self.edges.discard({u, v})
        self.neighbors[u].discard(v)
        self.neighbors[v].discard(u)
        self.degrees[u] -= 1
        self.degrees[v] -= 1

    def __sub__(self, v_or_edge):
        """ Deletes edge (x, y). or vertex (x): G-{u,v} or G - v
        """
        if isinstance(v_or_edge, int):
            self.remove_vertex(v_or_edge)
        elif len(v_or_edge) == 1:
            self.remove_vertex(v_or_edge[0])
        elif len(v_or_edge) == 2:
            self.remove_edge(*v_or_edge)

    def has_edge(self, x, y):
        """ Returns 1 if (x,y) is an edge in G"""
        return {x, y} in self.edges


    def neighborhood(self, x):
        """ Returns the vertices which are connected to v. """
        return self.neighbors[x]

    def N(self, x):
        """ Returns the vertices which are connected to v. """
        return self.neighbors[x]

    def order(self):
        """ Returns order i.e., the number of vertices """
        return len(self.vertices)


    def size(self):
        """ Returns size i.e., the number of edge """
        return len(self.edges)

    def degree(self, v):
        """ Returns the degree of the vertex v """
        return len(self.neighbors[v])


    def is_isolated_vertex(self, v):
        """ Returns 1 if the vertex v is isolated"""
        return len(self.neighbors[v]) == 0

    def is_planar(self, e = None, set_embedding = True):
        """ test if graph is planar
        """
        if e:
            if isinstance(e[0], int): e = [e]
            for x, y in e: self.add_edge(x, y)
            Q = is_planar(self, set_embedding = False)
            for x, y in e: self.remove_edge(x, y)
            return bool(Q)

        return bool(is_planar(self, set_embedding))

    def face_list(self):
        """
        List the faces of Graph (returned as a list of lists of edges (tuples) of
        the current embedding.

        """
        if not self.embedding:
            self.is_planar()  # get an embedding

        # Establish set of possible edges
        edgeset = set(map(tuple, self.edges))
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
                faces_sets.append(set(map(frozenset, path)))
                path = [edgeset.pop()]
            else:
                path.append(tup)
                edgeset.discard(tup)

        if len(path):
            faces_sets.append(set(map(frozenset, path)))

        return faces_sets

    def faces_iterator(self):
        """
        iterate over the faces of Graph (returned as a list of lists of edges (tuples) of
        the current embedding.

        """
        if not self.embedding:
            self.is_planar(None, True)  # assure an embedding

        # Establish set of possible edges
        edgeset = set(map(tuple, self.edges))
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

        D = self.__class__()
        for i in range(len(f)):
            D.add_vertex()

        vertex_faces = defaultdict(set)
        AddEdge = D.add_edge

        for i in range(len(f)):
            for v in f[i]:
                vertex_faces[v].add(i + 1)

        for i, j in combinations(range(len(f)), 2):
            if f_edges[i] & f_edges[j]:
                AddEdge(i + 1, j + 1)

        D.faces_sets = f_edges
        D.vertex_faces = vertex_faces

        return D

    def find_planarity_preserving_edge(self):
        """
        In a planar graph G, this function returns two vertices u,v
        such that the graph G remains planar after adding new edge uv to G 
        """

        for f in self.faces_iterator():
            if len(f) == 3: continue  # not neccesary but for speed up
            u = f[0]
            for v in f[2:-1]:
                if {u, v} not in self.edges:
                    return u, v

        return None

    def planarity_preserving_edges(self):
        """
        Returns a set of planarity preserving edges.
        """
        ppe = set()
        for f in random_permutation(self.faces_iterator()):
            if len(f) == 3: continue  # not necessary but for speed up
            u = f[0]
            for v in f[2:-1]:
                if {u, v} not in self.edges:
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
        return 100.0 * (len(self.edges) / (3.0 * len(self.vertices) - 6.0))

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
        E = set()
        N = self.neighbors
        for z in N[y] - N[x] - {x}:
            self.add_edge(x, z)
            E.add(frozenset((x, z)))

        self.identities[x] |= self.identities[y]
        self.remove_vertex(y)
        self.ncontractions += 1
        return E

    def __div__(self, e):
        self.contract(*e)

    def __truediv__(self, e):
        self.contract(*e)

    def __idiv__(self, e):
        self.contract(*e)

    def __itruediv__(self, e):
        self.contract(*e)

    def subdivide(self, x, y):
        """
        subdivide edge xy and return new vertex z between x and y
        """
        z = self.add_vertex()
        self.add_edge(x, z)
        self.add_edge(y, z)
        self.remove_edge(x, y)
        return z

    def is_connected(self):
        N = len(self.vertices)
        n_connected_vertices = 1
        V = set()
        v = random.sample(self.vertices, 1)
        V |= self.neighbors[v]
        while V:
            u = V.pop()
            n_connected_vertices += 1
            V |= self.neighbors[u]


    def is_complete(self, v_list = None):
        if v_list is None:
            return len(self.edges) == (len(self.vertices) * (len(self.vertices) - 1)) / 2.0

        for x, y in combinations(v_list, 2):
            if {x, y} not in self.edges: return False

        return True

    def is_regular(self, r = None):
        r = self.degree(self.neighbors.keys()[0]) if r is None else r
        for v in self.vertices:
            if self.degree(v) != r: return False

        return True

    def is_independent_set(self, v_list):
        for x, y in combinations(v_list, 2):
            if {x, y} in self.edges: return False

        return True

    def triangles(self):
        N = self.neighbors
        for x, y in self.edges:
            for z in N[x] & N[y]:
                yield x, y, z

    def four_clique(self):
        N = self.neighbors
        for x, y, z in self.triangles():
            for w in N[x] & N[y] & N[z]:
                return 1, [x, y, z, w]
        return 0, [None, None, None, None]

    def cliques_iter(self, k):
        """ very unefficient version of find_clique finding
        """
        N = self.neighbors
        rem_vertices = deque([({v}, N[v]) for v in self.vertices if len(N[v]) >= k - 1])
        append = rem_vertices.append
        pop = rem_vertices.pop

        while rem_vertices:

            vertices, nhood = pop()
            L = len(vertices)
            if L >= k: yield vertices, nhood

            nhood = {v for v in nhood if len(N[v] & nhood) >= k - 1 - L}
            if len(nhood) + L < k: break

            for v in nhood:
                append((vertices | {v} , N[v] & nhood))



    def find_clique(self, k):
        """ very unefficient version of clique finding
        """
        N = self.neighbors
        rem_vertices = deque([({v}, N[v]) for v in self.vertices if len(N[v]) >= k - 1])
        append = rem_vertices.append
        pop = rem_vertices.pop

        while rem_vertices:

            vertices, nhood = pop()
            L = len(vertices)
            if L >= k: return vertices, nhood

            nhood = {v for v in nhood if len(N[v] & nhood) >= k - 1 - L}
            if len(nhood) + L < k: return 0, 0

            for v in nhood:
                append((vertices | {v} , N[v] & nhood))

        return None, None






