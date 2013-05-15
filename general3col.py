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
import time
import random
from itertools import combinations
from coloring_tools import is_2colorable

# from Gato.Gred import DrawMyGraph

#-------------------------------------------------------------------------------
# Witness generation
#-------------------------------------------------------------------------------
opcodes = ['K4', 'K112', 'T31', 'C4', 'K3 Free', 'NOTP', 'MAXP', 'E']

def decode_operation(O, prefix = '', recursive_level = 1):


    if prefix != '': prefix = '  ' * recursive_level + prefix + ' '
    op = opcodes[O[0]]
    if   op == 'K4':      return prefix + 'Q.E.D. K4 ' + str(O[1])
    elif op == 'K112':    return prefix + 'K112 ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + ' ' + str(sorted(O[3])) + ' ' + str(sorted(O[4])) + '\n'
    elif op == 'T31':     return prefix + 'T31 ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + '\n' + '  ' * recursive_level + 'BEGIN\n' + UNCOL_witness(O[3], 'SUBT31', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'
    elif op == 'C4':      return prefix + 'C4 ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + '\n' + '  ' * recursive_level + 'BEGIN\n' + UNCOL_witness(O[3], 'SUBC4', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'
    elif op == 'E':       return prefix + 'E ' + str(sorted(O[1])) + '\n' + '  ' * recursive_level + 'BEGIN\n' + UNCOL_witness(O[3], 'SUBE', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'
    elif op == 'K3 Free': return prefix + 'Q.E.D. K3 free' + '\n'
    elif op == 'NOTP':    return prefix + 'G is not planar, please verify' + '\n'



def UNCOL_witness(P, prefix = '', recursive_level = 1):
    pf_text = ''
    values = sorted(P.values(), key = lambda x: x[-1])
    for o in values:
        pf_text += decode_operation(o, prefix, recursive_level)

    return pf_text

def COL_witness(G):
    C = dict()
    for color, v in enumerate(G.identities.iterkeys(), 1):
        C[color] = sorted(G.identities[v])

    str_col = repr(C)
    replace_tuples = ((',', ',\n'), (']', '\n]'), ('[', '\n[\n '), ('{', '{\n'), ('}', '\n}'))
    for older, newer in replace_tuples: str_col = str_col.replace(older, newer)

    return str_col

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Begining of the Algorithm
#-------------------------------------------------------------------------------

def find_non_edge2(G):
    """ iterate over at least P4 subgraphs $P_4$
    """
    N = G.adjLists
    V = G.vertices
    for x in sorted(V, key = G.degree, reverse = True):
        for y in sorted(G.vertices - N[x] - set([x]), key = lambda v: len(N[x] ^ N[v]), reverse = False):  # max common vertices, (N[x] | N[y])
            yield (x, y)

def find_non_edge(G):
    """ iterate over at least P4 subgraphs $P_4$
    """
    N = G.adjLists
    V = G.vertices
    for x in sorted(V, key = G.degree, reverse = True):
        for y in N[x]:
            for z in sorted(N[y] - N[x] - set([x]), key = lambda v: len(N[x] & N[v]), reverse = False):  # max common vertices, (N[x] | N[y])
                yield (x, z)

def find_non_edge3(G):
    """ iterate over at least P4 subgraphs $P_4$
    """
    N = G.adjLists
    V = G.vertices
    for x in sorted(V, key = G.degree, reverse = True):
        for y in N[x]:
            for z in sorted(N[y] - N[x] - set([x]), key = lambda v: len(N[x] & N[v]), reverse = True):
                for w in sorted(N[z] - N[x] - set([x, y]), key = lambda v: len(N[x] & N[v]), reverse = True):
                    yield (x, w)


def K112(K112, G, P):
    x, y, z, w = K112
    if set([z, w]) in G.edges:
        P['QED'] = [0, K112, len(P)]
        return 0, G, P

    P[max(z, w)] = [1, (z, w), (x, w), (y, w), K112, len(P)]
    G.contract(min(z, w), max(z, w))  # also deletes vertex max(z,w)
    return 1, G, P

def solve_K112(G, P):
    O = G.order
    find_K112 = G.find_K112

    N = O() + 1
    while O() < N:
        N = O()
        Q, XYZW = find_K112()
        if not Q: return 2, G, P
        Q, G, P = K112(XYZW, G, P)
        if not Q: return 0, G, P

    return 2, G, P

def solve_T31(G, P, alpha):
    for x, y, z, w in G.find_T31():
        H = G.copy()
        H.contract(min(x, w), max(x, w))
        Q, H, Pr = is_3colorable(H, alpha, False, False)
        if not Q:
            P[max(y, w)] = [2, (w, y), [x, y, z, w], Pr, len(P)]
            G.contract(min(y, w), max(y, w))
            return G, P

    return G, P

def solve_non_edge(G, P, alpha):
    for x, z in find_non_edge(G):
        H = G.copy()
        H.add_edge(x, z)
        Q, H, Pr = is_3colorable(H, alpha, False, False)
        if not Q:
            P[max(x, z)] = [opcodes.index('E'), (x, z), None, Pr, len(P)]
            G.contract(min(x, z), max(x, z))
            return G, P

    return G, P

def solve_C4(G, P, recursive_deep):
    for x, y, z, w in G.C4_iterator():
        H = G.copy()
        H.contract(min(x, z), max(x, z))
        Q, H, Pr = is_3colorable(H, recursive_deep, False, False)
        if not Q:
            P[max(y, w)] = [3, (w, y), [x, y, z, w], Pr, len(P)]
            G.contract(min(y, w), max(y, w))
            return G, P

    return G, P

def try_greedy(G):
    """
    This algorithm tries to color the remaining graph with a greedy algorithm
    using a different sarting order (a different triangle) for each step.
    It is just an accelerator algorithm is is stricly not needed.
    """
    V = G.vertices
    for x, y, z in G.triangles():
        H = G.copy()
        contract = H.contract
        E = H.edges
        for w in V - set([x, y, z]):
            if   set([x, w]) not in E: contract(x, w)
            elif set([y, w]) not in E: contract(y, w)
            elif set([z, w]) not in E: contract(z, w)
            else: break
        if H.order() <= 3: return 1, H

    return 0, G


def basic_tests(G, planar_test = True, K4_test = True, triangle_test = True):

    # request for a planargraph
    if planar_test:
        if not G.is_planar():  return 0, G, {'QED':[5, 'G is not planar', 0]}

    # request for a K4 free graph
    if K4_test:
        fcq , K4 = G.four_clique()
        if fcq: return 0, G, {'QED':[0, sorted(K4), 0]}

    # request for a graph with triangles
    if triangle_test:
        if G.is_triangle_free(): return 2, G, {'QED':[4, 0]}

    return None, G, None


def is_3colorable(G, alpha = 1, try_greedy_coloring = False, perform_basic_tests = True, planar_test = True, K4_test = True, triangle_test = True):

    if perform_basic_tests:
        Q, G, P = basic_tests(G, planar_test, K4_test, triangle_test)
        if Q is not None:
            return Q, G, P

    P = dict()
    N = G.order() + 1

    while G.order() < N:

        N = G.order()
        Q, G, P = solve_K112(G, P)
        if not Q: return 0, G, P
        if G.order() <= 3: return 1, G, P
        if alpha:
            # L = G.order()
            # G, P = solve_T31(G, P, alpha - 1)
            # if G.order()==L:
            G, P = solve_non_edge(G, P, alpha - 1)

    if try_greedy_coloring:
        Q, G = try_greedy(G)
        if Q:
            return 1, G, P

    return 3, G, P

#-----------------------------------------------------------------------------------------------

def planar_3COL(G, recursive_deep = 1):

    Q, H, P = is_3colorable(G.copy(), recursive_deep, try_greedy_coloring = True, perform_basic_tests = True)
    if Q in (0, 2): return Q, G, P
    if Q == 1: return 1, H, P

    N = G.order() + 1
    M = 0

    G.is_planar(None, True)  # update embedding
    ppe = G.get_planar_preserving_edges()

    while len(ppe):

        e = ppe.pop()
        x, y = min(e), max(e)
        if set([x, y]) in G.edges: continue
        if not set([x, y]) <= G.vertices: continue

        H = G.copy()
        G.contract(x, y)

        # test contraction G/x,y
        Q, Gout, P = is_3colorable(G.copy(), recursive_deep, try_greedy_coloring = True, perform_basic_tests = False)

        if Q == 1: return 1, Gout, P  # 3-coloring found

        if not Q:  # contraction failed, hence add edge xy
            G = H.copy()  # restore G from original
            G.add_edge(x, y)


        if len(G) <= 3: return 1, G, dict()




    # print 'triangulation', G.is_planar(), G.order(), G.size(), 3 * G.order() - 6
    if G.size() != (3 * G.order()) - 6:
        raise RuntimeError("there is a problem, seems to be a triangulation but does not have 3V-6 edges")

    # planar triangulation
    # A maximal planar graph is 3 colorable iff all its vertex degrees are even
    for x in G.vertices:
        if G.degree(x) % 2:
            return 0, G, {'QED':[6, "G is a maximal planar graph with an odd vertex \n" + str((G.is_planar(), G.order(), G.size(), 3 * G.order() - 6)), 0]}

    Q, G, P = is_3colorable(G, 0, True, False)  # if it is a triangulation, try_greedy_coloring finds a 3-coloring quickly!
    return 1, G, dict()

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

def general_3COL(G, recursive_deep = 1):
    # request for a K4 free graph
    fcq , K4 = G.four_clique()
    if fcq:
        return 0, G, {'QED':[0, sorted(K4), 0]}

    Q, G, P = is_3colorable(G, recursive_deep, try_greedy_coloring = True, perform_basic_tests = False)
    if Q in (0, 1): return Q, G, P
    G_out = G.copy()

    x = max(G.vertices, key = G.degree)
    while G.degree(x) < len(G) - 1:
        # y = max(G.vertices - G[x] - set([x]), key = lambda v: len( G[x] ^ G[v] ))
        y = max(G.vertices - G[x] - set([x]), key = lambda v: len(G[v] | G[x]))
        H = G.copy()

        G.contract(x, y)
        # test contraction G/x,y
        Q, G, P = is_3colorable(G, recursive_deep, try_greedy_coloring = True, perform_basic_tests = False)
        if Q == 1: return 1, G, P  # 3-coloring found
        if not Q:  # contraction failed, hence add edge xy
            G = H.copy()  # restore G from original
            G.add_edge(x, y)

        if len(G) <= 3: return 1, G, dict()
        x = max(G.vertices, key = G.degree)


    # try a 2-coloring of G[x]
    H = is_2colorable(G, x)
    if H is not None:
        return 1, H, dict()


    return 3, G_out, dict()


#-----------------------------------------------------------------------------------------------

def incremental_deep_3COL(G, max_recursive_deep = 10):
    for recursive_deep in range(max_recursive_deep + 1):
        Q, G, P = general_3COL(G.copy(), recursive_deep)
        if Q in (0, 1): return Q, G, P, recursive_deep


    return 3, G, dict(), max_recursive_deep + 1






