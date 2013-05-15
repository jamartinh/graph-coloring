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
from itertools import combinations



#-------------------------------------------------------------------------------
# Backtracking 3-coloring algorithm
#-------------------------------------------------------------------------------
def is_bipartite(G, V = None, A = None, B = None):
    """Returns a two-coloring of the graph.
    """
    if V is None: V = G.vertices
    elif not isinstance(V, set): V = set(V)
    if A is None: A = set()
    if B is None: B = set()
    A, B = set(A), set(B)
    E = G.edges

    for x in V:  # handle disconnected graphs
        if x in A | B: continue
        queue = [x]

        if   all(({x, y} not in E for y in A)): A.add(x)
        elif all(({x, y} not in E for y in B)): B.add(x)
        else: return None

        while queue:
            v = queue.pop()
            C = B if v in A else A  # opposite color of node v
            for w in G[v] & V:  # consider only vertices contained in V
                if w in A | B:
                    if {w, v} <= A or {w, v} <= B:  # if w,v are in the same partition
                        return None
                else:
                    C.add(w)
                    queue.append(w)
    return A, B


def is_2colorable(G, u):
    """ try a 2-coloring of G[u]
        u: is a complete vertex and so the graph will be 3-colored
    """

    H = G.copy()
    H.remove_vertex(u)
    contract = H.contract
    AB = is_bipartite(H)
    if AB is not None:
        A, B = AB
        x = A.pop()
        for y in A: contract(x, y)
        x = B.pop()
        for y in B: contract(x, y)

        H.add_named_vertex(u)
        H.identities[u] = G.identities[u]
        return H

    return None


def is_clique(G, s):
    k = len(s)
    for v in s: s.intersection_update(G[v] | {v})
    return len(s) == k



def is_independent_set(G, s):
    for x, y in combinations(s, 2):
        if {x, y} in G.edges: return False
    return True


def verify_colorig(C, G):
    for i in range(3):
        if not is_independent_set(G, C[i]):
            return False

    return True


def greedy_coloring(G):
    V = lambda G: G.vertices
    E = lambda G: G.edges
    N = G.neighborhood
    while len(E(G)) < (len(G) * (len(G) - 1)) / 2.0:
        u = min(V(G), key = G.degree)
        v = min((V(G) - N(u) - {u}), key = G.degree)
        G.contract(u, v)

    return G


def maximal_independent_sets(G, R = None, P = None, X = None):
    if R is None: R = set()
    if P is None: P = set()
    if X is None: X = set()
    """
    bors kerbosch maximal cliques algorithm
    """
    coN = lambda x: G.vertices - G[x] - {x}

    if len(P) == 0 and len(X) == 0:
        if len(R) > 0:
            yield sorted(R)
        return


    # d, pivot = max([(len(coN[v]), v) for v in P | X])
    pivot = max(P | X, key = lambda v: len(coN[v]))

    for v in P - coN[pivot]:
        for x in maximal_independent_sets(G, R | {v} , P & coN[v], X & coN[v]):
            yield x
        P.discard(v)
        X.add(v)


def bt_coloring(n, i, v_list, C, G):
    """
    3^N brute force algorithm for 3 coloring  with backtracking and degree
    vertex ordering (higher first) heuristics
    """
    if i > n: return 1, C

    for j in range(3):
# #        if j==0:  A,B = C[1],C[2]
# #        if j==1:  A,B = C[0],C[2]
# #        if j==2:  A,B = C[0],C[1]
# #        if not is_bipartite(G, G[v_list[i]], A, B):
# #            continue

        if all(({u, v_list[i]} not in G.edges for u in C[j])):
            x = map(list, C)
            x[j].append(v_list[i])
            Q, x = bt_coloring(n, i + 1, v_list, x, G)
            if Q: return 1, x

    return 0, [[], [], []]

def is_3colorable_BF(G):
    """
    brute force backtracking coloring algorithm
    """

    v_list = sorted(G.Vertices(), key = G.degree, reverse = True)

    Q, _K4 = G.four_clique()
    if Q: return 0, [[], [], []], None

    Q, C = bt_coloring(len(v_list) - 1, 0, v_list, [[], [], []], G)
    return Q, C, None

def bt_coloring_witness(color_list):
    C = dict()
    for color in (0, 1, 2):
        C[color + 1] = sorted(color_list[color])

    str_col = repr(C)
    replace_tuples = ((',', ',\n'), (']', '\n]'), ('[', '\n[\n '), ('{', '{\n'), ('}', '\n}'))
    for older, newer in replace_tuples: str_col = str_col.replace(older, newer)

    return str_col
