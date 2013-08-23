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
from coloring_tools import is_2colorable
# from planegraphs import set_copy
from itertools import combinations


# constant
UNDETERMINED = 3
#-------------------------------------------------------------------------------
# Witness generation
#-------------------------------------------------------------------------------
Operation_Codes = {'K4': 'A complete graph on 4 vertices',
                   'K112': 'A diamond graph',
                   'T31': 'A T_{31} sub graph',
                   'C4': 'a square subraph, a cycle of length four',
                   'K3 Free': 'The graph is trangle free',
                   'NOTP': 'The graph is not planar',
                   'MAXP': 'maximal planar graph',
                   'E': 'A contraction due to a simple non-edge is implicit identity',
                   'SYM': ' contraction due to a vertex neighborhood subsumtion'
                    }



def decode_operation(O, prefix = '', recursive_level = 1):

    if prefix != '':
        prefix = '  ' * recursive_level + prefix + ' '

    op = O[0]
    if op == 'K4':
        return prefix + 'K4 ' + str(O[1]) + '\n' + prefix + 'Q.E.D.'
    elif op == 'K112':
        return prefix + 'K112 ' + str(tuple(sorted(O[1]))) + '\n'
    elif op == 'SYM':
        return prefix + 'SYM ' + str(tuple(sorted(O[1]))) + '\n'
    elif op == 'T31':
        return prefix + 'T31 ' + str(tuple(sorted(O[1]))) + ' ;' + str(sorted(O[2])) + '\n' + '  ' * recursive_level + 'BEGIN SUB T31\n' + '  ' * (recursive_level + 1) + 'TRY CONTRACTION ' + (O[2][0], O[2][3]) + '\n' + UNCOL_witness(O[3], ' ', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'
    elif op == 'C4':
        return prefix + 'C4: ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + '\n' + '  ' * recursive_level + 'BEGIN SUB C4\n' + UNCOL_witness(O[3], ' ', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'
    elif op == 'E':
        return prefix + 'E ' + str(tuple(sorted(O[1]))) + '\n' + '  ' * recursive_level + 'BEGIN SUB E\n' + '  ' * (recursive_level + 1) + 'TRY NEW EDGE ' + str(tuple(sorted(O[1]))) + '\n' + UNCOL_witness(O[3], ' ', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'
    elif op == 'K3 Free':
        return prefix + 'Triangle free planar graph.\n Q.E.D.' + '\n'
    elif op == 'NOTP':
        return prefix + 'G is not planar, please verify' + '\n'


def UNCOL_witness(P, prefix = '', recursive_level = 1):
    pf_text = ''
    values = sorted(P.values(), key = lambda x: x[-1])
    for o in values:
        pf_text += decode_operation(o, prefix, recursive_level)

    return pf_text


def COL_witness(G, P = None):
    if P: return UNCOL_witness(P)

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


def find_T31(G):
    """ iterate over tadpole subgraphs $T_{3,1}$
    """
    N = G.neighbors
    V = G.vertices
    for x in sorted(V, key = G.degree, reverse = True):
        for w in sorted(V - N[x] - {x}, key = lambda v: len(N[x] ^ N[v]), reverse = True):
            for y in N[x] - {w}:
                for z in N[x] & N[y] & N[w]:
                    yield (x, y, z, w)


# alternative version
def T31_iterator(G):
    """ iterate over tadpole subgraphs $T_{3,1}$
    """
    N = G.neighbors
    for x, y, z in G.triangles():
        for w in N[z] - {x, y} - N[x] - N[y]: yield x, y, z, w
        for w in N[y] - {x, z} - N[x] - N[x]: yield x, z, y, w
        for w in N[x] - {y, z} - N[y] - N[z]: yield z, y, x, w


def find_C4(G):
    """ iterate over C4 subgraph $C_4$
    """
    N = G.neighbors
    V = G.vertices
    for x in V:
        for y in N[x]:
            for z in N[y] - N[x] - {x}:
                for w in (N[x] & N[z]) - N[y]:
                    yield (x, y, z, w)


def find_K112(G):
    """ Finds a 3-partite complete subgraph $K_{1,1,2}$
    """
    N = G.neighbors
    for x, y in G.edges:
        z = N[x] & N[y]
        if len(z) > 1: return 1, x, y, sorted(z)

    return 0, None, None, None


def K112(x, y, z, w, G, P):

    if {z, w} in G.edges:
        # there is a K4 subgraph
        P['QED'] = ['K4', (x, y, z, w), len(P)]
        return 0, G, P

    z, w = (z, w) if z < w else (w, z)
    P[w] = ('K112', (z, w), len(P))
    G.contract(z, w)  # also deletes vertex w
    return 1, G, P


def solve_K112(G, P):
    while True:
        Q, x, y, zw_list = find_K112(G)
        if not Q: return 2, G, P
        z = zw_list.pop(0)
        for w in zw_list:
            Q, G, P = K112(x, y, z, w, G, P)
            if not Q: return 0, G, P

    return 2, G, P



def solve_T31(G, P, alpha, test_fun):
    for x, y, z, w in find_T31(G):
        H = G.copy()
        x, w = (x, w) if x <= w else (w, x)
        H.contract(x, w)
        Q, H, Pr = test_fun(H, alpha)
        if not Q:
            P[max(y, w)] = ('T31', (w, y), (x, y, z, w), Pr, len(P))
            G.contract(min(y, w), max(y, w))
            return True, G, P

    return False, G, P



# too slow to be useful
def find_symmetric_vertices(G, P):
    """
    An improved heuristic: if some vertex y subsumes the neighborhood of another vertex x
    it is safe, for the 3-colorability test, to contract them. 
    """
    N = G.neighbors
    V = G.vertices
    for x, y in combinations(V, 2):
        if N[x] <= N[y] or N[y] <= N[x]:
            P[max(x, y)] = ('SYM', (x, y), None, len(P))
            G.contract(min(x, y), max(x, y))
            return True, G, P

    return False, G, P


def find_non_edge(G):
    """ iterate over at least P4 subgraphs $P_4$
    """
    N = G.neighbors
    V = G.vertices
    for x in sorted(V, key = G.degree, reverse = True):
    # for x in sorted(V, key = G.degree, reverse = True):
        for y in N[x]:
            for z in sorted(N[y] - N[x] - {x}, key = lambda v: len(N[x] & N[v]),
                            reverse = False):  # max common vertices, (N[x] | N[y])
                yield (x, z)




def solve_non_edge2(G, P, alpha):
    """
    Two possible choices: 
    a) try a vertex contraction. If it fails, add a new edge. Or
    b) Try adding a new edge. If it fails, contract the vertices.
    
    """
    for e in find_non_edge(G):
        H = G.copy()
        x, y = min(e), max(e)
        H.contract(x, y)
        Q, H, Pr = is_3colorable(H, alpha)
        if not Q:
            P[(x, y)] = ['E', (x, y), None, Pr, len(P)]
            G.add_edge(x, y)
            return True, G, P

    return False, G, P


def solve_non_edge(G, P, alpha):
    """
    Two possible choices: 
    a) try a vertex contraction. If it fails, add a new edge. Or
    b) Try adding a new edge. If it fails, contract the vertices.
    
    """
    for e in find_non_edge(G):
        H = G.copy()
        x, y = min(e), max(e)
        H.add_edge(x, y)
        Q, H, Pr = is_3colorable(H, alpha)
        if not Q:
            P[(x, y)] = ['E', (x, y), None, Pr, len(P)]
            H.contract(x, y)
            return True, H, P

    return False, G, P


def basic_tests(G, planar_test = True, triangle_test = True):
    # request for a planar graph
    if planar_test:
        if not G.is_planar():  return 0, G, {'QED': [5, 'G is not planar', 0]}

    # request for a graph with triangles
    if triangle_test:
        if G.is_triangle_free(): return 2, G, {'QED': [4, 0, 0]}

    return None, G, None


def is_3colorable_plane(G, alpha = 1, planar_test = True, triangle_test = True):
    """
    Attempts to find a 3-uncolorability certificate for a planar graph
    """
    if planar_test or triangle_test:
        Q, G, P = basic_tests(G, planar_test, triangle_test)
        if Q is not None:
            return Q, G, P

    P = dict()
    N = G.order() + 1

    while G.order() < N:
        N = G.order()

        Q, G, P = solve_K112(G, P)
        if not Q: return 0, G, P
        # Q, G, P = find_symmetric_vertices(G, P)
        # if Q: continue
        if G.order() <= 3: return 1, G, P
        if alpha:
            Q, G, P = solve_T31(G, P, alpha - 1, is_3colorable_plane)

    return UNDETERMINED, G, P


def is_3colorable(G, alpha = 1):
    """
    Attempts to find a 3-uncolorability certificate for graph
    """
    P = dict()
    N = G.order() + 1
    while G.order() < N:
        N = G.order()
        Q, G, P = solve_K112(G, P)
        if not Q: return 0, G, P
        # Q, G, P = find_symmetric_vertices(G, P)
        # if Q: continue
        if G.order() <= 3: return 1, G, P
        if alpha:
            Q, G, P = solve_T31(G, P, alpha - 1, is_3colorable)
            if Q or G.order() < N: continue
            Q, G, P = solve_non_edge(G, P, alpha - 1)
            # if Q or G.order() < N: continue
            # else: break

    return UNDETERMINED, G, P

#-----------------------------------------------------------------------------------------------

def planar_3COL(G, alpha = 1):
    """
    Tries to find a 3-coloring of a planar graph G.
    """

    Q, H, P = is_3colorable_plane(G.copy(), alpha, planar_test = True, triangle_test = True)
    if Q in (0, 2): return Q, G, P
    if Q == 1: return 1, H, P

    G_out = G.copy()
    G.is_planar(None, True)  # update embedding
    ppe = G.planarity_preserving_edges()

    while len(ppe):

        e = ppe.pop()
        x, y = min(e), max(e)

        # check if a previous contraction or edge addition has deleted a planarity-preserving-edge
        if {x, y} in G.edges or not {x, y} <= G.vertices: continue

        H = G.copy()
        G.contract(x, y)

        # test contraction G/x,y
        Q, G, P = is_3colorable_plane(G, alpha, planar_test = False, triangle_test = False)
        if Q == 1: return 1, G, P  # 3-coloring found
        if not Q:  # contraction failed, hence add edge xy
            G = H.copy()  # restore G from original
            G.add_edge(x, y)

        if len(G) <= 3: return 1, G, dict()

    Q, G, P = is_3colorable_plane(G, 0, False, False)  # if it is a triangulation, solve_K112 finds a 3-coloring quickly!
    if Q == 1:
        return 1, G, dict()

    return UNDETERMINED, G_out, dict()


#-----------------------------------------------------------------------------------------------

def general_3COL(G, alpha = 1):
    """
    Tries to find a 3-coloring of a graph G.
    """
    Q, G, P = is_3colorable(G, alpha)
    if Q in (0, 1): return Q, G, P
    G_out = G.copy()

    u = max(G.vertices, key = G.degree)


    while G.degree(u) < G.order() - 1:

        # heuristic selection
        v = max(G.vertices - G[u] - {u}, key = lambda v: len(G[v] | G[u]))
        H = G.copy()

        G.contract(u, v)
        # test contraction G/u,v
        Q, G, P = is_3colorable(G, alpha)
        if Q == 1: return 1, G, P  # 3-coloring found
        if not Q:  # contraction failed, hence add edge xy
            G = H.copy()  # restore G from original
            G.add_edge(u, v)

        if G.order() <= 3: return 1, G, dict()

        u = max(G.vertices, key = G.degree)

    # try a 2-coloring of G[u]
    H = is_2colorable(G, u)
    if H is not None:
        return 1, H, dict()

    return UNDETERMINED, G_out, dict()


#-----------------------------------------------------------------------------------------------
# automatic algorithms
#-----------------------------------------------------------------------------------------------
def incremental_depth_planar_3COL(G, max_alpha = 10):
    for alpha in range(max_alpha + 1):
        Q, G, P = planar_3COL(G.copy(), alpha)
        if Q in (0, 1): return Q, G, P, alpha

    return UNDETERMINED, G, dict(), max_alpha + 1


def incremental_depth_3COL(G, max_alpha = 10):
    for alpha in range(max_alpha + 1):
        Q, G, P = general_3COL(G.copy(), alpha)
        if Q in (0, 1): return Q, G, P, alpha

    return UNDETERMINED, G, dict(), max_alpha + 1






