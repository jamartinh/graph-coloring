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

"""
This module provides functions to find cliques in graphs.

The general method is based on a proof system.


The proof system is as follows:

Let G be a simple graph:
A k-clique of G is a subset H of G of cardinality k and such that all the vertices of H have degree k.
So, if there is a k-clique in G, then there is a sequence D of vertex deletions such that G-D = Kk. 

1. Axiom: a simple proof of the non-existence of a k-clique is a set of (V-k-1)-vertices of degree(v)<k.
Thais is, there are less than k-vertices of degree at least k (an anti clique of size k).

2. A simplification rule:
    If by deleting an edge uv there is an anti clique of size k, then all the non-common neighbors of u and v
    sould be removed from the graph, since u,v must form part of any possible k-clique of G.

3. An alpha-bounded research rule (a free asumption):
    While alpha >0: 
        delete an arbitrary edge u,v
        if there is no application of (1) or (2)
            apply (3) for alpha=alpha-1
        else:
            apply (1) or (2)
        until finding (1)

"""



from itertools import combinations
from math import floor
import sys
import time

#-------------------------------------------------------------------------------
# Witness generation
#-------------------------------------------------------------------------------
opcodes = ['A', 'SV', 'SE', 'R']
UNDETERMINED = 3

def decode_operation(O, k, prefix = '', recursive_level = 1):


    if prefix != '': prefix = '  ' * recursive_level + prefix + ' '
    op = opcodes[O[0]]
    if   op == 'A':    return prefix + 'Q.E.D. anti clique: ' + str(O[1]) + ' of size ' + str(len(O[1])) + ' found. \n'
    # elif op == 'SV':    return prefix + 'Simplify: remove vertices: ' + str(sorted(O[1])) + ' for d(v) <= (k-1) =' + str(k-1) + '\n'
    elif op == 'SV':    return prefix + 'Simplify: remove vertices of d(v) <= (k-1), for k =' + str(k) + '\n'
    elif op == 'SE':    return prefix + 'Simplify: remove edges u,v of |N(u).N(v)| <= (k-2), for k =' + str(k) + '\n'
    # elif op == 'SE':    return prefix + 'Simplify: remove edges: ' + str(sorted(O[1])) + ' for k=' + str(k) + '\n'
    elif op == 'R':    return prefix + 'Assume: ' + str(sorted(O[1])) + '\n' + '  ' * recursive_level + 'BEGIN\n' + NO_witness(O[3], k, 'SUB ', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'




def NO_witness(P, k, prefix = '', recursive_level = 1):
    """   
    """
    pf_text = ''
    values = sorted(P.values(), key = lambda x: x[-1])
    for o in values:
        pf_text += decode_operation(o, k, prefix, recursive_level)

    return pf_text

def YES_witness(G, k, P = None):

    if P: return NO_witness(P, k)


    C = dict()
    for color, v in enumerate(G.identities.iterkeys(), 1):
        C[color] = sorted(G.identities[v])

    str_col = repr(C)
    replace_tuples = ((',', ',\n'), (']', '\n]'), ('[', '\n[\n '), ('{', '{\n'), ('}', '\n}'))
    for older, newer in replace_tuples: str_col = str_col.replace(older, newer)

    return str_col

#-------------------------------------------------------------------------------
# Begining of the Algorithm
#-------------------------------------------------------------------------------

def filter_vertices_by_degree(G, k):
    """
    Return the vertices of G of degree at leat k.
    """
    # return {v for v in G.vertices if G.degrees[v] >= k - 1}
    return  {v for v in G.vertices if len(G.neighbors[v]) >= k - 1}


def filter_edges_by_degree(G, k):
    """
    Return the edges of G of degree less than  k-2.
    """
    return  {(u, v) for u, v in G.edges if len(G.neighbors[u] & G.neighbors[v]) < k - 2}



def axiom(V, k, P):
    if len(V) < k:
        P['QED'] = [opcodes.index('A'), V, len(P)]
        return 0, P

    return 2, P



def simplify(G, k, P):

    while True:

        # Axiom
        # Q, P = axiom(G.vertices, k, P)
        # if not Q: return 0, G, P
        if len(G.vertices) < k:
            P['QED'] = [opcodes.index('A'), G.vertices, len(P)]
            return 0, G, P


        V = filter_vertices_by_degree(G, k)
        #V = {v for v in G.vertices if len(G.neighbors[v]) >= k - 1}
        if len(V) < len(G.vertices):  # or len(E) < len(G.edges):
            G = G.subgraph(V)
            P[frozenset(V)] = [opcodes.index('SV'), V, len(P)]
            continue

#       E = filter_edges_by_degree(G, k)
#       # E = {(u, v) for u, v in G.edges if len(G.neighbors[u] & G.neighbors[v]) < k - 2}
#        if len(E):
#            for u, v in E: G.remove_edge(u, v)
#            P[E] = [opcodes.index('SE'), E, len(P)]
#            continue

        break

    return 2, G, P





def research(G, k, P, alpha):
    """
    Do research by removing an edge and see what hapends.
    """
    for u, v in sorted(G.edges, key = lambda e: len(G[min(e)] ^ G[max(e)])):
    # for u in sorted(G.vertices, key = G.degree, reverse = True):
        # if {u, v} not in G.edges or not {u, v} <= G.vertices: continue

        D = (G[u] & G[v]) | {u, v}
        H = G.subgraph(D)
        Q, H, Pr = has_kclique(H, k, alpha)
        if not Q:
            G.remove_edge(u , v)
            P[(u, v)] = [opcodes.index('R'), (u, v), None, Pr, len(P)]
            return G, P

    return G, P


def has_kclique(G, k, alpha = 0):
    """
    Search for a proof of non-k-clique up to alpha decision level.
    """

    if len(G) >= k and G.is_complete(): return 1, G, dict()

    P = dict()
    N = len(G) + 1
    M = len(G.edges) + 1
    while len(G) < N or len(G.edges) < M:
        N = len(G)
        M = len(G.edges)

        Q, G, P = simplify(G, k, P)
        if not Q: return 0, G, P

        if len(G) >= k and G.is_complete(): return 1, G, P

        if alpha:
            G, P = research(G, k, P, alpha - 1)




    return UNDETERMINED, G, P


#-----------------------------------------------------------------------------------------------


def kclique(G, k, alpha = 0):

    Q, G, P = has_kclique(G, k, alpha)
    if Q in (0, 1): return Q, G, P

    G_out = G.copy()
    cur_clique = set()
    while len(cur_clique) < k and not G.is_complete() and len(G):

        u = max(G.vertices - cur_clique, key = G.degree)

        # save a copy
        # H = G.copy()

        # test reduction
        D = G[u] | {u}
        H = G.subgraph(D)
        Q, H, P = has_kclique(H, k, alpha)
        if Q == 1: return 1, H, P
        if not Q:  # failed, hence remove edge uv
            G.remove_vertex(u)
        else:
            G = H
            cur_clique.add(u)

        D = filter_vertices_by_degree(G, k)
        G = G.subgraph(D)

        print "advance:", len(cur_clique), len(G)

    G = G_out.subgraph(cur_clique)
    if len(G) >= k and G.is_complete():  return 1, G, P

    return UNDETERMINED, G_out, dict()
#-----------------------------------------------------------------------------------------------
# automatic algorithms
#-----------------------------------------------------------------------------------------------

def E(G): return G.edges
def V(G): return G.vertices


def greedy_clique(G):
    assigned = set()
    u = max(G.vertices, key = G.degree)
    while not G.is_complete():
        v = max(G[u] - assigned, key = lambda v: len(G[u] & G[v]))
        D = (G[u] & G[v]) | {u, v}
        G = G.subgraph(D)
        assigned.add(v)

    return len(G), G


def max_clique(G, alpha = 5):

    a = 3
    b, H = greedy_clique(G.copy())
    print b, "clique by greedy"
    last_good = b
    k = int(floor((a + b) / 2))
    while k != a:
        print "a, b, c", a, k, b
        H = G.copy()
        Q, H, _P = kclique(H, k, alpha)
        if Q == 0:
            a = k + 1
            print "no", k, "clique"
        elif Q == 1:
            b = k
            last_good = k
            print k, "clique"
        else: return last_good, G, {}, last_good
        k = int(floor((a + b) / 2))

    Q, H, P = kclique(H, last_good - 1, alpha)
    return last_good, H, P, last_good

