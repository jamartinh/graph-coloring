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
import random
from itertools import combinations, ifilter
from planegraphs import Graph, set_copy, pairwise, probabilistic_choice


#-------------------------------------------------------------------------------
# Special Graphs
#-------------------------------------------------------------------------------

def complete_graph(N):
    G = Graph()
    G + N  # add N vertices

    for u, v in combinations(G.Vertices(), 2):
        G.add_edge(u, v)

    return G



#-------------------------------------------------------------------------------
# Random Graphs
#-------------------------------------------------------------------------------


def planar_density(N, M):
    return round(100.0 * (M / (3.0 * N - 6.0)), 2)

def edges_from_density(N, D):
    return int(round((D * (3.0 * N - 6.0)) / 100))

def graph_connectivity(N, M):
    # return N * M / (N * (N - 1.0) / 2.0)
    return 2.0 * M / float(N)

def edges_from_connectivity(N, c = 4.67):
    # return int(round(c * (N - 1.0) / 2.0))
    return int(round(c * N / 2.0))

def get_maximal_planar_edges(G, n, direction):
    """this fuction is from GATO ToolBox"""

    Edges = list()  # 6*n
    AdjEdges = dict()
    v_list = list()
    for v in G.vertices:
        AdjEdges[v] = list()
        v_list.append(v)

    index = 0
    a = v_list[index]
    index += 1
    b = v_list[index]
    index += 1

    Edges.append((a, b))
    AdjEdges[a].append((a, b))
    Edges.append((b, a))
    AdjEdges[b].append((b, a))

    m = 2
    while index < n:
        e = Edges[random.randint(0, m - 1)]
        v = v_list[index]
        index = index + 1

        while e[1] != v:
            x = (v, e[0])
            Edges.append(x)
            m = m + 1
            AdjEdges[v].append(x)

            y = (e[0], v)
            Edges.append(y)
            m = m + 1
            AdjEdges[e[0]].insert(AdjEdges[e[0]].index(e) + 1, y)

            index2 = AdjEdges[e[1]].index((e[1], e[0]))
            if index2 == 0:
                e = AdjEdges[e[1]][-1]
            else:
                e = AdjEdges[e[1]][index2 - 1]

    if direction == 0:  # undirected
        m = m - 1
        while m > 0:
            del Edges[m]
            m = m - 2

    return Edges



def random_planar_graph(n, m):
    G = Graph()

    for _v in xrange(n):
        G.add_vertex()

    Edges = get_maximal_planar_edges(G, n, 0)

    for _i in xrange(m):
        pos = random.randint(0, len(Edges) - 1)
        G.add_edge(Edges[pos][0], Edges[pos][1])
        del Edges[pos]

    return G


def random_planar_g(N, M):
    # ensure a pseudo-random planar graph with "low" probability of K4
    G = random_planar_graph(N, M)
    nmax = 100
    counter = 1

    Q, _K4 = G.four_clique()

    while Q and counter < nmax:
        G = random_planar_graph(N, M)
        counter += 1
        Q, _K4 = G.four_clique()

    return G


def random_briggs_graph(n, m):
    """Return the random graph G_{n,m}.

    Gives a graph picked randomly out of the set of all graphs
    with n nodes and m edges.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.

    Notes
    -----
    Algorithm by Keith M. Briggs Mar 31, 2006.
    Inspired by Knuth's Algorithm S (Selection sampling technique),
    in section 3.4.2 of

    References
    ----------
    .. [1] Donald E. Knuth,
        The Art of Computer Programming,
        Volume 2 / Seminumerical algorithms
        Third Edition, Addison-Wesley, 1997.

    """
    mmax = n * (n - 1) / 2
    if m >= mmax:
        G = complete_graph(n)
    else:
        G = Graph()
        G + n

    if n == 1 or m >= mmax:
        return G

    u = 0
    v = 1
    t = 0
    k = 0
    while True:
        if random.randrange(mmax - t) < m - k:
            G.add_edge(u, v)
            k += 1
            if k == m: return G
        t += 1
        v += 1
        if v == n:  # go to next row of adjacency matrix
            u += 1
            v = u + 1


def random_connected_graph(n, m):
    """ Return the random connected graph G_{n,m}.

    Gives a graph picked randomly out of the set of all graphs
    with n nodes and m edges.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.

    """
    G = Graph()
    V = G + n  # add n vertices

    max_edges = int((n * (n - 1.0)) / 2.0)
    m = min(m, max_edges)


    # add the first connection line, (n-1) edges, assuring a connected graph
    for u, v in pairwise(V):
        G.add_edge(u, v)

    AddEdge = G.add_edge
    E_star = set_copy(combinations(G.vertices, 2))

    for u, v in random.sample(E_star - G.edges, m - n + 1):
        AddEdge(u, v)

    return G


#-------------------------------------------------------------------------------
# 4-regular planar graphs
#-------------------------------------------------------------------------------

def phi_A(G):
    """
    two nonadjacent edges in the same face
    N increases by 1
    """

    f = next(ifilter(lambda f: len(f) > 3, G.faces_iterator()), None)
    if not f: return G

    i = random.randint(0, len(f) - 1)
    d = f[i]
    a = f[(i + 1) % len(f)]

    v = range(i + 2, i + 2 + len(f) - 1)
    for j in random.sample(v, len(v)):
        b = f[ j % len(f) ]
        c = f[ (j + 1) % len(f) ]
        if set([a, d]) & {b, c}: continue
        break

    y = G.add_vertex()
    for x in (a, b, c, d):
        G.add_edge(x, y)

    G.del_edge(a, d)
    G.del_edge(b, c)

    return G

def phi_B(G):
    """
    for each triangle
    N increases by 2
    """
    # find a triangular face (not the same as finding a triangle)
    f = next(ifilter(lambda f: len(f) == 3, G.faces_iterator()), None)
    if not f: return G
    a, b, u = f

    emb_u = G.embedding[u]

    # if next of b is not a swap(a,b)
    if emb_u[(emb_u.index(b) + 1) % len(emb_u)] != a:
        a, b = b, a

    d = emb_u[(emb_u.index(a) + 1) % len(emb_u)]
    c = emb_u[(emb_u.index(d) + 1) % len(emb_u)]

    G.del_vertex(u)
    x = G.subdivide(a, b)
    y, z = G.add_vertex(), G.add_vertex()

    G.add_edge(x, y)
    G.add_edge(x, z)
    G.add_edge(y, z)

    G.add_edge(z, a)
    G.add_edge(z, d)

    G.add_edge(y, b)
    G.add_edge(y, c)

    return G

def phi_C(G):
    """
    for each vertex
    N increases by 4
    """
    u = random.sample(G.Vertices(), 1)
    u = u[0]  # u is actually a 1-element list!


    w, v, y, x = (G.subdivide(u, i) for i in G.embedding[u])

    for i, j in pairwise((w, v, y, x, w)):
        G.add_edge(i, j)


    return G

def phi_F(G):
    """
    two square faces with a common edge
    N increases by 2
    """
    for fa, fb in combinations(G.faces_iterator(), 2):
        if len(fa) != 4 or len(fb) != 4: continue
        b_y = set(fa) & set(fb)
        if len(b_y) != 2: continue
        b, y = b_y

        # check that the succesor of b is y otherwise swap faces
        if fa[(fa.index(b) + 1) % len(fa)] != y:
            fb, fa = fa, fb

        # check that in face fa the sucesor of b is y and in face fb the sucesor
        # of y is b other wise find other face
        if fa[(fa.index(b) + 1) % len(fa)] != y or fb[(fb.index(y) + 1) % len(fb)] != b: continue

        x = fa[(fa.index(y) + 1) % len(fa)]
        a = fa[(fa.index(x) + 1) % len(fa)]

        c = fb[(fb.index(b) + 1) % len(fb)]
        z = fb[(fb.index(c) + 1) % len(fb)]

        v = G.subdivide(b, y)
        w = G.subdivide(v, y)

        G.add_edge(a, v)
        G.add_edge(c, v)

        G.add_edge(x, w)
        G.add_edge(z, w)

        G.del_edge(a, x)
        G.del_edge(c, z)

        break


    return G



def octahedron():
    G = Graph()

    v, w, x, y, z1, z2 = (G.add_vertex() for i in xrange(6))

    for i, j in pairwise((v, w, x, y, v)):
        G.add_edge(i, j)
        G.add_edge(i, z1)
        G.add_edge(i, z2)

    G.is_planar()
    return G


def random_4regular_planar_graph(N, G = None):
    if G is None: G = octahedron()
    else: G.is_planar(None, True)
    f_list = (phi_A, phi_B, phi_C, phi_F)
    prob_f = (.80  , .05  , .10  , .05)
    # prob_f = (.48  , .02  , .48  , .02)
    # prob_f = (.25  , .25  , .25  , .25)

    while G.order() < N:

        f = probabilistic_choice(f_list, prob_f)
        G = f(G)
        if not G.is_planar(): raise RuntimeError("non-planar error ")


    if not G.is_regular(4): raise RuntimeError("non-regular error ")
    return G

def iter_random_4regular_planar_graphs(N_low, N_top, G = None):
    if G is None: G = octahedron()
    else: G.is_planar(None, True)  # asures an embedding

    f_list = (phi_A, phi_B, phi_C, phi_F)
    prob_f = (.25  , .25  , .25  , .25)

    while G.order() < N_top:

        f = probabilistic_choice(f_list, prob_f)
        V = G.order()
        G = f(G)
        if V != G.order() and V > N_low:
            yield G

def recursive_4regular_planar_graphs(N, p, G = None):
    if G is None: G = octahedron()
    func = (phi_A, phi_B, phi_C, phi_F)
    for f in func:
        if G.order() < N:
            H = G.copy()
            Gf = f(H)
            if Gf.order() != H.order():
                for out_G in recursive_4regular_planar_graphs(N, p, Gf):
                    yield out_G


