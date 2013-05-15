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
from itertools import combinations
from math import floor
import sys
import time

#-------------------------------------------------------------------------------
# Witness generation
#-------------------------------------------------------------------------------
opcodes = ['Kk', 'Kk-', 'E']
UNDETERMINED = 3

def decode_operation(O, k, prefix = '', recursive_level = 1):


    if prefix != '': prefix = '  ' * recursive_level + prefix + ' '
    op = opcodes[O[0]]
    if   op == 'Kk':     return prefix + 'Q.E.D. K' + str(k) + ' found ' + str(O[1])
    elif op == 'Kk-':    return prefix + 'K' + str(k) + '- ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + ' ' + str(sorted(O[3])) + ' ' + str(sorted(O[4])) + '\n'
    elif op == 'E':      return prefix + 'E ' + str(sorted(O[1])) + '\n' + '  ' * recursive_level + 'BEGIN\n' + UNCOL_witness(O[3], k, 'SUBE', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'




def UNCOL_witness(P, k, prefix = '', recursive_level = 1):
    pf_text = ''
    values = sorted(P.values(), key = lambda x: x[-1])
    for o in values:
        pf_text += decode_operation(o, k, prefix, recursive_level)

    return pf_text

def COL_witness(G, k, P = None):

    if P: return UNCOL_witness(P, k)


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

def _find_non_edge5(G):
    """ iterate over at least P4 subgraphs $P_4$.
    """
    N = G.neighbors
    V = G.vertices
    for x in sorted(V, key = G.degree, reverse = True):
        for y in N[x]:
            for z in sorted(N[y] - N[x] - {x}, key = lambda v: len(N[x] & N[v]), reverse = False):  # max common vertices, (N[x] | N[y])
                yield (x, z)

def _find_non_edge(G):
    for u in G.vertices:
        for v in G.vertices - G[u] - {u}:
            yield (u, v)

    # u = min(G.vertices, key = G.degree)
    # v = max(G.vertices - G[u] - {u}, key = lambda v: len(G[u] & G[v]))

def find_non_edge(G):
    return ((u, v)  for u, v in combinations(G.vertices, 2) if {u, v} not in G.edges)


    # u = min(G.vertices, key = G.degree)
    # v = max(G.vertices - G[u] - {u}, key = lambda v: len(G[u] & G[v]))

def find_kclique_minus(G, k):
    for clique, nhood in G.cliques_iter(k):
        if len(nhood) > 1:
            return clique, nhood
    return None, None

def kclique_minus(found_clique, z, w, G, P):
    if {z, w} in G.edges:
        P['QED'] = [0, found_clique | {z, w}, len(P)]
        return 0, G, P
    z, w = (z, w) if z < w else (w, z)
    # P[max(z, w)] = (1, (z, w), (x, w), (y, w), K112, len(P))
    P[z] = (1, found_clique , {z, w}, len(P))
    G.contract(z, w)  # also deletes vertex max(z,w)
    return 1, G, P

def solve_kclique_minus(G, k, P):
    O = G.order
    N = O() + 1
    while O() < N:
        N = O()
        found_clique, nhood = find_kclique_minus(G, k)
        if not found_clique: return 2, G, P
        z = min(nhood)
        for w in nhood - {z}:
            Q, G, P = kclique_minus(found_clique, z, w, G, P)
            if not Q: return 0, G, P

    return 2, G, P




def research(G, k, P, alpha):
    """
    Do research by adding an edge and see what happends.
    If adding an edge leads to a Kk, then there is an unavoidable vertex contraction.
    """
    # print "enter research *****************"
    for u, v in combinations(G.vertices, 2):
        # print "for", (u, v)
        if {u, v} in G or not {u, v} <= G.vertices: continue
        H = G.copy()
        H.add_edge(u, v)
        Q, H, Pr = is_kcolorable(H, k, alpha)
        if not Q:
            P[max(u, v)] = [opcodes.index('E'), (u, v), None, Pr, len(P)]
            G.contract(min(u, v), max(u, v))
            # print "contract", u, v, len(G), "alpha", alpha
            # print "exit research *****************"
            return G, P
    # print "enter research *****************"
    return G, P





def is_kcolorable(G, k, alpha = 0):
    """
    Search for a proof of non-k-colorability up to alpha decision level.
    """
    if G.order() <= k: return 1, G, dict()

    P = dict()
    N = G.order() + 1

    while G.order() < N:
        N = G.order()

        Q, G, P = solve_kclique_minus(G, k, P)
        if not Q: return 0, G, P
        if G.order() <= k: return 1, G, P

        if alpha:
            G, P = research(G, k, P, alpha - 1)

    return UNDETERMINED, G, P


#-----------------------------------------------------------------------------------------------


def kcoloring(G, k, alpha = 0):

    # request for a KK free graph
    found_clique, _nhood = G.find_clique(k + 1)
    if found_clique: return 0, G, {'QED':[0, found_clique, 0]}

    Q, G, P = is_kcolorable(G, k, alpha)
    if Q in (0, 1): return Q, G, P
    G_out = G.copy()

    u = min(G.vertices, key = G.degree)
    while len(G) > k and G.degree(u) < len(G) - 1:

        v = max(G.vertices - G[u] - {u}, key = lambda v: len(G[u] & G[v]))

        # save a copy
        H = G.copy()
        # test contraction G/x,y
        G.contract(u, v)
        Q, G, P = is_kcolorable(G, k, alpha)
        if Q == 1: return 1, G, P
        if not Q:  # contraction failed, hence add edge uv
            G = H.copy()  # restore G from original
            G.add_edge(u, v)

        if len(G) <= k: return 1, G, dict()
        u = min(G.vertices, key = G.degree)


    return UNDETERMINED, G_out, dict()


#-----------------------------------------------------------------------------------------------
# automatic algorithms
#-----------------------------------------------------------------------------------------------

def E(G): return G.edges
def V(G): return G.vertices


def greedy_col(G):
    N = G.neighbors
    while len(G.edges) < (len(G) * (len(G) - 1)) / 2:
        u = min(G.vertices, key = G.degree)
        v = min(G.vertices - {u} - N[u], key = G.degree)
        u, v = sorted((u, v))
        G.contract(u, v)

    return len(G), G


def chi_coloring(G, alpha = 5):

    a = 3
    b, H = greedy_col(G.copy())
    print b, "colorable by greedy"
    last_good = b
    k = int(floor((a + b) / 2))
    while k != a:
        print "a, b, c", a, k, b
        H = G.copy()
        Q, H, _P = kcoloring(H, k, alpha)
        if Q == 0:
            a = k + 1
            print "no", k, "colorable"
        elif Q == 1:
            b = k
            last_good = k
            print k, "colorable"
        else: return last_good, G, {}, last_good
        k = int(floor((a + b) / 2))

    Q, H, P = kcoloring(H, last_good - 1, alpha)
    return last_good, H, P, last_good

#-------------------------------------------------------------------------------------------
# TESTING
#-------------------------------------------------------------------------------------------
def main(strFileName, alpha = 5):
    import graphio as gio
    import testutil as tu
    print " *** Starting *** "
    G = gio.load_from_edge_list_named(strFileName)

    V, E, avgdeg = G.order(), G.size(), G.avg_degree()
    t1 = time.clock()
    k, G, P, last_good = chi_coloring(G, alpha)
    Q = True
    # k = 5
    # x, y = G.find_clique(k)
    # print "find_clique", sorted(x)
    # Q, P = True, None
    # Q, G, P = kcoloring(G, k , alpha = 2)
    # Q, G, P = is_kcolorable(G, k , alpha = 2)
    t2 = time.clock()
    strScreen = "Q: %s k: %d N: %d M: %d time1: %3.3f average degree: %2.2f " % (Q, k, V, E, t2 - t1, avgdeg)
    print strScreen
    witness = COL_witness(G, k, P)
    # if not Q:  witness = r3.UNCOL_witness(P)
    # else: witness = r3.COL_witness(G, P)

    tu.save_witness("", k, witness, "", strFileName)


if __name__ == "Main":
    main(sys.argv[1], sys.argv[2])


# main("to_3coloring_uf50-0955.cnf.col", 5)
main("kinstances/queen6_6.col", 2)


