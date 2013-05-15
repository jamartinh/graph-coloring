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
#
#    author: Alexander Schliep (alexander@schliep.org)
#   Recent modifications by Jose Antonio Martin H. 17/05/2011
#       Copyright (C) 1998-2010, Alexander Schliep, Winfried Hochstaettler and
#       Copyright 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
#
#       Contact: alexander@schliep.org, winfried.hochstaettler@fernuni-hagen.de
#
#       Information: http://gato.sf.net


###############################################################################
###############################################################################
###############################################################################
#                                                                             #
#                            AN IMPLEMENTATION OF                             #
#                           THE HOPCROFT AND TARJAN                           #
#                    PLANARITY TEST AND EMBEDDING ALGORITHM                   #
#                                                                             #
###############################################################################
#                                                                             #
# References:                                                                 #
#                                                                             #
# [Meh84] K.Mehlhorn.                                                         #
#         "Data Structures and Efficient Algorithms."                         #
#         Springer Verlag, 1984.                                              #
# [MM94]  K.Mehlhorn and P.Mutzel.                                            #
#         "On the embedding phase of the Hopcroft and Tarjan planarity        #
#          testing algorithm."                                                #
#         Technical report no. 117/94, mpi, Saarbruecken, 1994                #
#                                                                             #
###############################################################################

from copy import deepcopy
import random
class Stack:
    """ Simple Stack class implemented as a Python list """


    def __init__(self):
        self.contents = []

    def Push(self, v):
        self.contents.append(v)

    def Pop(self):
        v = self.contents[-1]
        self.contents = self.contents[:-1]
        return v

    def Clear(self):
        self.contents = []

    def IsEmpty(self):
        return (len(self.contents) == 0)

    def IsNotEmpty(self):
        return (len(self.contents) > 0)

    def Contains(self, v):
        return v in self.contents




#-------------------------------------------------------------------------------
# PlanarityTest
#-------------------------------------------------------------------------------

#=============================================================================#
class List:

    def __init__(self, el = []):
        elc = deepcopy(el)
        self.elements = elc

        # a) Access Operations
    def length(self):
        return len(self.elements)

    def empty(self):
        if self.length() == 0:
            return 1
        else:
            return 0

    def head(self):
        return self.elements[0]

    def tail(self):
        return self.elements[-1]

        # b)Update Operations
    def push(self, x):
        self.elements.insert(0, x)
        return x

    def Push(self, x):
        self.elements.append(x)
        return x

    def append(self, x):
        self.Push(x)

    def pop(self):
        x = self.elements[0]
        self.elements = self.elements[1:]
        return x

    def Pop(self):
        x = self.elements[-1]
        self.elements = self.elements[:-1]
        return x

    def clear(self):
        self.elements = []

    def conc(self, A):
        self.elements = self.elements + A.elements
        A.elements = []
        #=============================================================================#



        #=============================================================================#
class pt_graph:


    def __init__(self):
        self.V = []
        self.E = []
        self.adjEdges = {}

        # a) Access operations
    def source(self, e):
        return e[0]

    def target(self, e):
        return e[1]

    def number_of_nodes(self):
        return len(self.V)

    def number_of_edges(self):
        return len(self.E)

    def all_nodes(self):
        return self.V

    def all_edges(self):
        return self.E

    def adj_edges(self, v):
        return self.adjEdges[v]

    def adj_nodes(self, v):
        nodelist = []
        for e in self.adj_edges(v):
            nodelist.append(e[1])
        return nodelist

    def first_node(self):
        return self.V[0]

    def last_node(self):
        return self.V[-1]

    def first_edge(self):
        return self.E[0]

    def last_edge(self):
        return self.E[-1]

    def first_adj_edge(self, v):
        if len(self.adj_edges(v)) > 0:
            return self.adj_edges(v)[0]
        else:
            return None

    def last_adj_edge(self, v):
        if len(self.adj_edges(v)) > 0:
            return self.adj_edges(v)[-1]
        else:
            return None

            # b) Update operations
    def new_node(self, v):
        self.V.append(v)
        self.adjEdges[v] = []
        return v

    def new_edge(self, v, w):
        if v == w:  # Loop
            raise KeyError
        if (v, w) in self.E:  # Multiple edge
            raise KeyError
        self.E.append((v, w))
        self.adjEdges[v].append((v, w))
        return (v, w)

    def del_node(self, v):
        try:
            for k in self.V:
                for e in self.adj_edges(k):
                    if source(e) == v or target(e) == v:
                        self.adjEdges[k].remove(e)
            self.V.remove(v)
            for e in self.E:
                if source(e) == v or target(e) == v:
                    self.E.remove(e)
        except KeyError:
            raise KeyError

    def del_edge(self, e):
        try:
            self.E.remove(e)
            self.adjEdges[source(e)].remove((source(e), target(e)))
        except KeyError:
            raise KeyError

    def del_nodes(self, node_list):  # deletes all nodes in list L from self

        L = deepcopy(node_list)
        for l in L:
            self.del_node(l)

    def del_edges(self, edge_list):  # deletes all edges in list L from self

        L = deepcopy(edge_list)
        for l in L:
            self.del_edge(l)

    def del_all_nodes(self):  # deletes all nodes from self
        self.del_nodes(self.all_nodes())

    def del_all_edges(self):  # deletes all edges from self
        self.del_edges(self.all_edges())

    def sort_edges(self, cost):

        sorted_list = cost.items()
        sorted_list.sort(key = lambda x: x[1])
        self.del_all_edges()
        for i in sorted_list:
            self.new_edge(source(i[0]), target(i[0]))

def source(e):
    return e[0]

def target(e):
    return e[1]

def reversal(e):
    return (e[1], e[0])
    #=============================================================================#



    #=============================================================================#
class block:
# The constructor takes an edge and a list of attachments and creates
# a block having the edge as the only segment in its left side.
#
# |flip| interchanges the two sides of a block.
#
# |head_of_Latt| and |head_of_Ratt| return the first elements
# on |Latt| and |Ratt| respectively
# and |Latt_empty| and |Ratt_empty| check these lists for emptyness.
#
# |left_interlace| checks whether the block interlaces with the left
# side of the topmost block of stack |S|.
# |right_interlace| does the same for the right side.
#
# |combine| combines the block with another block |Bprime| by simply
# concatenating all lists.
#
# |clean| removes the attachment |w| from the block |B| (it is
# guaranteed to be the first attachment of |B|).
# If the block becomes empty then it records the placement of all
# segments in the block in the array |alpha| and returns true.
# Otherwise it returns false.
#
# |add_to_Att| first makes sure that the right side has no attachment
# above |w0| (by flipping); when |add_to_Att| is called at least one
# side has no attachment above |w0|.
# |add_to_Att| then adds the lists |Ratt| and |Latt| to the output list
# |Att| and records the placement of all segments in the block in |alpha|.




    def __init__(self, e, A):
        self.Latt = List(); self.Ratt = List()  # list of attachments "ints"
        self.Lseg = List(); self.Rseg = List()  # list of segments represented by
        # their defining "edges"
        self.Lseg.append(e)
        self.Latt.conc(A)  # the other two lists are empty

    def flip(self):


        ha = List()  # "ints"
        he = List()  # "edges"

        # we first interchange |Latt| and |Ratt| and then |Lseg| and |Rseg|
        ha.conc(self.Ratt); self.Ratt.conc(self.Latt); self.Latt.conc(ha);
        he.conc(self.Rseg); self.Rseg.conc(self.Lseg); self.Lseg.conc(he);

    def head_of_Latt(self):
        return self.Latt.head()

    def empty_Latt(self):
        return self.Latt.empty()

    def head_of_Ratt(self):
        return self.Ratt.head()

    def empty_Ratt(self):
        return self.Ratt.empty()

    def left_interlace(self, S):
        # check for interlacing with the left side of the
        # topmost block of |S|
        if (S.IsNotEmpty() and not((S.contents[-1]).empty_Latt()) and
            self.Latt.tail() < (S.contents[-1]).head_of_Latt()):
            return 1
        else:
            return 0

    def right_interlace(self, S):
        # check for interlacing with the right side of the
        # topmost block of |S|
        if (S.IsNotEmpty() and not((S.contents[-1]).empty_Ratt()) and
            self.Latt.tail() < (S.contents[-1]).head_of_Ratt()):
            return 1
        else:
            return 0

    def combine(self, Bprime):
        # add block Bprime to the rear of |this| block
        self.Latt.conc(Bprime.Latt)
        self.Ratt.conc(Bprime.Ratt)
        self.Lseg.conc(Bprime.Lseg)
        self.Rseg.conc(Bprime.Rseg)
        Bprime = None

    def clean(self, dfsnum_w, alpha, dfsnum):
        # remove all attachments to |w|; there may be several
        while not(self.Latt.empty()) and self.Latt.head() == dfsnum_w:
            self.Latt.pop()
        while not(self.Ratt.empty()) and self.Ratt.head() == dfsnum_w:
            self.Ratt.pop()
        if not(self.Latt.empty()) or not(self.Ratt.empty()):
            return 0

            # |Latt| and |Ratt| are empty;
            #  we record the placement of the subsegments in |alpha|.
        for e in self.Lseg.elements:
            alpha[e] = left
        for e in self.Rseg.elements:
            alpha[e] = right
        return 1

    def add_to_Att(self, Att, dfsnum_w0, alpha, dfsnum):
        # add the block to the rear of |Att|. Flip if necessary
        if not(self.Ratt.empty()) and self.head_of_Ratt() > dfsnum_w0:
            self.flip()
        Att.conc(self.Latt)
        Att.conc(self.Ratt)
        # This needs some explanation.
        # Note that |Ratt| is either empty or {w0}.
        # Also if |Ratt| is non-empty then all subsequent
        # sets are contained in {w0}.
        # So we indeed compute an ordered set of attachments.
        for e in self.Lseg.elements:
            alpha[e] = left
        for e in self.Rseg.elements:
            alpha[e] = right
            #=============================================================================#



            #=============================================================================#
            # GLOBALS:

left = 1
right = 2
G = pt_graph()

reached = {}
dfsnum = {}
parent = {}
dfs_count = 0
lowpt = {}
Del = []
lowpt1 = {}
lowpt2 = {}
alpha = {}
Att = List()
cur_nr = 0
sort_num = {}
tree_edge_into = {}
#=============================================================================#



#=============================================================================#
def planarity_test(Gin, set_embedding = True):
# planarity_test decides whether the InputGraph is planar.
# it also order the adjecentLists in counterclockwise.



    n = Gin.order()  # number of nodes
    if n < 3: return 1
    if Gin.size() > 3 * n - 6:  return 0  # number of edges
    if Gin.size() > 6 * n - 12: return 0

    #--------------------------------------------------------------
    # make G a copy of Gin and make G bidirected

    global G, cur_nr
    G = pt_graph()

    for v in Gin.vertices:
        G.new_node(v)
    for e in Gin.edges:
        e = list(e)
        G.new_edge(source(e), target(e))

    cur_nr = 0
    nr = {}
    cost = {}
    n = G.number_of_nodes()
    for v in G.all_nodes():
        nr[v] = cur_nr
        cur_nr = cur_nr + 1
    for e in G.all_edges():
        if nr[source(e)] < nr[target(e)]:
            cost[e] = n * nr[source(e)] + nr[target(e)]
        else:
            cost[e] = n * nr[target(e)] + nr[source(e)]
    G.sort_edges(cost)

    L = List(G.all_edges())
    while not(L.empty()):
        e = L.pop()
        if (not(L.empty()) and source(e) == target(L.head())
            and source(L.head()) == target(e)):
            L.pop()
        else:
            G.new_edge(target(e), source(e))
            #--------------------------------------------------------------


            #--------------------------------------------------------------
            # make G biconnected
    Make_biconnected_graph()
    #--------------------------------------------------------------


    #--------------------------------------------------------------
    # make H a copy of G
    #
    # We need the biconnected version of G (G will be further modified
    # during the planarity test) in order to construct the planar embedding.
    # So we store it as a graph H.
    H = deepcopy(G)
    #--------------------------------------------------------------


    #--------------------------------------------------------------
    # test planarity

    global dfsnum, parent, alpha, Att

    dfsnum = {}
    parent = {}
    for v in G.all_nodes():
        parent[v] = None

    reorder()

    alpha = {}
    for e in G.all_edges():
        alpha[e] = 0
    Att = List()
    alpha[G.first_adj_edge(G.first_node())] = left

    if not(strongly_planar(G.first_adj_edge(G.first_node()), Att)):
        return 0
        #--------------------------------------------------------------


        #--------------------------------------------------------------
        # construct embedding

    global sort_num, tree_edge_into

    T = List()
    A = List()

    cur_nr = 0
    sort_num = {}
    tree_edge_into = {}

    embedding(G.first_adj_edge(G.first_node()), left, T, A)

    # |T| contains all edges incident to the first node except the
    # cycle edge into it.
    # That edge comprises |A|.
    T.conc(A)

    for e in T.elements:
        sort_num[e] = cur_nr
        cur_nr = cur_nr + 1

    H.sort_edges(sort_num)


    if not set_embedding:  return H.all_edges()

    # construct embedding:
    _embedding = dict()
    # print "embeding tests **************"
    for v in Gin.vertices:
        _embedding[v] = [y for x, y in reversed(H.adjEdges[v])]
    #    print v,_embedding[v]
    # print "***************"

    Gin.embedding = _embedding

    return H.all_edges()  # ccwOrderedEges



import itertools as it
# from networkx.algorithms import bipartite



def color(G):
    N = G.neighbors
    deg = G.degree

    color = {}
    for n in G.vertices:  # handle disconnected graphs
        if n in color or len(G[n]) == 0:  # skip isolates
            continue
        queue = [n]
        color[n] = 1  # nodes seen with color (1 or 0)
        while queue:
            v = queue.pop()
            c = 1 - color[v]  # opposite color of node v
            for w in N[v]:
                if w in color:
                    if color[w] == color[v]:
                        return None
                else:
                    color[w] = c
                    queue.append(w)
    # color isolates with 0
    color.update(dict.fromkeys([v for v in G.vertices if deg(v) == 0], 0))
    return color

def is_bipartite(G):
    if color(G): return True

    return False

def sets(G):
    """Returns bipartite node sets of graph G.

    Raises an exception if the graph is not bipartite.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    (X,Y) : two-tuple of sets
       One set of nodes for each part of the bipartite graph.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> X, Y = bipartite.sets(G)
    >>> list(X)
    [0, 2]
    >>> list(Y)
    [1, 3]

    See Also
    --------
    color
    """
    c = color(G)
    X = set(n for n in c if c[n])  # c[n] == 1
    Y = set(n for n in c if not c[n])  # c[n] == 0
    return (X, Y)

def is_planar2(G):

    """
    function checks if graph G has K(5) or K(3,3) as minors,
    returns True /False on planarity and nodes of "bad_minor"
    """
    result = True
    bad_minor = []
    n = len(G)
    if n > 5:
        for subnodes in it.combinations(G.vertices, 6):
            subG = G.subgraph(subnodes)
            if is_bipartite(G):  # check if the graph G has a subgraph K(3,3)
                X, _Y = sets(G)
                if len(X) == 3:
                    result = False
                    bad_minor = subnodes

    if n > 4 and result:
        for subnodes in it.combinations(G.vertices, 5):
            subG = G.subgraph(subnodes)
            if len(subG.edges) == 10:  # check if the graph G has a subgraph K(5)
                result = False
                bad_minor = subnodes
    if result: return map(list, G.edges)

    return False


def is_planar(G, set_embedding = True):
    # if not is_planar2(G): return False
    # if not set_embedding: return True
    return planarity_test(G, set_embedding)

#=============================================================================#


#=============================================================================#
def pt_DFS(v):


    global G, reached

    S = Stack()

    if reached[v] == 0:
        reached[v] = 1
        S.Push(v)

    while S.IsNotEmpty():
        v = S.Pop()
        for w in G.adj_nodes(v):
            if reached[w] == 0:
                reached[w] = 1
                S.Push(w)
#=============================================================================#


#=============================================================================#
def Make_biconnected_graph():
# We first make it connected by linking all roots of a DFS-forest.
# Assume now that G is connected.
# Let a be any articulation point and let u and v be neighbors
# of a belonging to different biconnected components.
# Then there are embeddings of the two components with the edges
# {u,a} and {v,a} on the boundary of the unbounded face.
# Hence we may add the edge {u,v} without destroying planarity.
# Proceeding in this way we make G biconnected.

    global G, reached, dfsnum, parent, dfs_count, lowpt

    #--------------------------------------------------------------
    # We first make G connected by linking all roots of the DFS-forest.
    reached = {}
    for v in G.all_nodes():
        reached[v] = 0
    u = G.first_node()

    for v in G.all_nodes():
        if  not(reached[v]):
            # explore the connected component with root v
            pt_DFS(v)
            if u != v:
                # link v's component to the first component
                G.new_edge(u, v)
                G.new_edge(v, u)
                #--------------------------------------------------------------


                #--------------------------------------------------------------
                # We next make G biconnected.
    for v in G.all_nodes():
        reached[v] = 0
    dfsnum = {}
    parent = {}
    for v in G.all_nodes():
        parent[v] = None
    dfs_count = 0
    lowpt = {}
    dfs_in_make_biconnected_graph(G.first_node())
    #--------------------------------------------------------------


    #=============================================================================#



    #=============================================================================#
def dfs_in_make_biconnected_graph(v):
# This procedure determines articulation points and adds appropriate
# edges whenever it discovers one.

    global G, reached, dfsnum, parent, dfs_count, lowpt

    dfsnum[v] = dfs_count
    dfs_count = dfs_count + 1
    lowpt[v] = dfsnum[v]
    reached[v] = 1

    if not(G.first_adj_edge(v)): return  # no children

    u = target(G.first_adj_edge(v))  # first child

    for e in G.adj_edges(v):
        w = target(e)
        if not(reached[w]):
            # e is a tree edge
            parent[w] = v
            dfs_in_make_biconnected_graph(w)
            if lowpt[w] == dfsnum[v]:
                # v is an articulation point. We now add an edge.
                # If w is the first child and v has a parent then we
                # connect w and parent[v], if w is a first child and v
                # has no parent then we do nothing.
                # If w is not the first child then we connect w to the
                # first child.
                # The net effect of all of this is to link all children
                # of an articulation point to the first child and the
                # first child to the parent (if it exists).
                if w == u and parent[v]:
                    G.new_edge(w, parent[v])
                    G.new_edge(parent[v], w)
                if w != u:
                    G.new_edge(u, w)
                    G.new_edge(w, u)

            lowpt[v] = min(lowpt[v], lowpt[w])

        else:
            lowpt[v] = min(lowpt[v], dfsnum[w])  # non tree edge
            #=============================================================================#



            #=============================================================================#
def reorder():
# The procedure reorder first performs DFS to compute dfsnum, parent
# lowpt1 and lowpt2, and the list Del of all forward edges and all
# reversals of tree edges.
# It then deletes the edges in Del and finally reorders the edges.

    global G, dfsnum, parent, reached, dfs_count, Del, lowpt1, lowpt2

    reached = {}
    for v in G.all_nodes():
        reached[v] = 0
    dfs_count = 0
    Del = []
    lowpt1 = {}
    lowpt2 = {}

    dfs_in_reorder(G.first_node())

    #--------------------------------------------------------------
    # remove forward and reversals of tree edges
    for e in Del:
        G.del_edge(e)
        #--------------------------------------------------------------


        #--------------------------------------------------------------
        # we now reorder adjacency lists
    cost = {}
    for e in G.all_edges():
        v = source(e)
        w = target(e)
        if dfsnum[w] < dfsnum[v]:
            cost[e] = 2 * dfsnum[w]
        elif lowpt2[w] >= dfsnum[v]:
            cost[e] = 2 * lowpt1[w]
        else:
            cost[e] = 2 * lowpt1[w] + 1
    G.sort_edges(cost)
    #--------------------------------------------------------------


    #=============================================================================#



    #=============================================================================#
def dfs_in_reorder(v):

    global G, dfsnum, parent, reached, dfs_count, Del, lowpt1, lowpt2

    #--------------------------------------------------------------
    dfsnum[v] = dfs_count
    dfs_count = dfs_count + 1
    lowpt1[v] = lowpt2[v] = dfsnum[v]
    reached[v] = 1
    for e in G.adj_edges(v):
        w = target(e)
        if not(reached[w]):
            # e is a tree edge
            parent[w] = v
            dfs_in_reorder(w)
            lowpt1[v] = min(lowpt1[v], lowpt1[w])
        else:
            lowpt1[v] = min(lowpt1[v], dfsnum[w])  # no effect for forward edges
            if dfsnum[w] >= dfsnum[v] or w == parent[v]:
                # forward edge or reversal of tree edge
                Del.append(e)
                #--------------------------------------------------------------


                #--------------------------------------------------------------
                # we know |lowpt1[v]| at this point and now make a second pass over all
                # adjacent edges of |v| to compute |lowpt2|
    for e in G.adj_edges(v):
        w = target(e)
        if parent[w] == v:
            # tree edge
            if lowpt1[w] != lowpt1[v]:
                lowpt2[v] = min(lowpt2[v], lowpt1[w])
            lowpt2[v] = min(lowpt2[v], lowpt2[w])
        else:
            # all other edges
            if lowpt1[v] != dfsnum[w]:
                lowpt2[v] = min(lowpt2[v], dfsnum[w])
                #--------------------------------------------------------------


                #=============================================================================#



                #=============================================================================#
def strongly_planar(e0, Att):
# We now come to the heart of the planarity test: procedure strongly_planar.
# It takes a tree edge e0=(x,y) and tests whether the segment S(e0) is
# strongly planar.
# If successful it returns (in Att) the ordered list of attachments of S(e0)
# (excluding x); high DFS-numbers are at the front of the list.
# In alpha it records the placement of the subsegments.
#
# strongly_planar operates in three phases.
# It first constructs the cycle C(e0) underlying the segment S(e0).
# It then constructs the interlacing graph for the segments emanating >from the
# spine of the cycle.
# If this graph is non-bipartite then the segment S(e0) is non-planar.
# If it is bipartite then the segment is planar.
# In this case the third phase checks whether the segment is strongly planar
# and, if so, computes its list of attachments.

    global G, alpha, dfsnum, parent

    #--------------------------------------------------------------
    # DETERMINE THE CYCLE C(e0)
    # We determine the cycle "C(e0)" by following first edges until a back
    # edge is encountered.
    # |wk| will be the last node on the tree path and |w0|
    # is the destination of the back edge.
    x = source(e0)
    y = target(e0)
    e = G.first_adj_edge(y)
    wk = y

    while dfsnum[target(e)] > dfsnum[wk]:  # e is a tree edge
        wk = target(e)
        e = G.first_adj_edge(wk)
    w0 = target(e)
    #--------------------------------------------------------------


    #--------------------------------------------------------------
    # PROCESS ALL EDGES LEAVING THE SPINE
    # The second phase of |strongly_planar| constructs the connected
    # components of the interlacing graph of the segments emananating
    # from the the spine of the cycle "C(e0)".
    # We call a connected component a "block".
    # For each block we store the segments comprising its left and
    # right side (lists |Lseg| and |Rseg| contain the edges defining
    # these segments) and the ordered list of attachments of the segments
    # in the block;
    # lists |Latt| and |Ratt| contain the DFS-numbers of the attachments;
    # high DFS-numbers are at the front of the list.
    #
    # We process the edges leaving the spine of "S(e0)" starting at
    # node |wk| and working backwards.
    # The interlacing graph of the segments emanating from
    # the cycle is represented as a stack |S| of blocks.
    w = wk
    S = Stack()

    while w != x:
        count = 0
        for e in G.adj_edges(w):
            count = count + 1

            if count != 1:  # no action for first edge
                # TEST RECURSIVELY
                # Let "e" be any edge leaving the spine.
                # We need to test whether "S(e)" is strongly planar
                # and if so compute its list |A| of attachments.
                # If "e" is a tree edge we call our procedure recursively
                # and if "e" is a back edge then "S(e)" is certainly strongly
                # planar and |target(e)| is the only attachment.
                # If we detect non-planarity we return false and free
                # the storage allocated for the blocks of stack |S|.
                A = List()
                if dfsnum[w] < dfsnum[target(e)]:
                    # tree edge
                    if not(strongly_planar(e, A)):
                        while S.IsNotEmpty(): S.Pop()
                        return 0
                else:
                    A.append(dfsnum[target(e)])  # a back edge

                    # UPDATE STACK |S| OF ATTACHMENTS
                    # The list |A| contains the ordered list of attachments
                    # of segment "S(e)".
                    # We create an new block consisting only of segment "S(e)"
                    # (in its L-part) and then combine this block with the
                    # topmost block of stack |S| as long as there is interlacing.
                    # We check for interlacing with the L-part.
                    # If there is interlacing then we flip the two sides of the
                    # topmost block.
                    # If there is still interlacing with the left side then the
                    # interlacing graph is non-bipartite and we declare the graph
                    # non-planar (and also free the storage allocated for the
                    # blocks).
                    # Otherwise we check for interlacing with the R-part.
                    # If there is interlacing then we combine |B| with the topmost
                    # block and repeat the process with the new topmost block.
                    # If there is no interlacing then we push block |B| onto |S|.
                B = block(e, A)

                while 1:
                    if B.left_interlace(S): (S.contents[-1]).flip()
                    if B.left_interlace(S):
                        B = None
                        while S.IsNotEmpty(): S.Pop()
                        return 0
                    if B.right_interlace(S): B.combine(S.Pop())
                    else: break
                S.Push(B)

                # PREPARE FOR NEXT ITERATION
                # We have now processed all edges emanating from vertex |w|.
                # Before starting to process edges emanating from vertex
                # |parent[w]| we remove |parent[w]| from the list of attachments
                # of the topmost
                # block of stack |S|.
                # If this block becomes empty then we pop it from the stack and
                # record the placement for all segments in the block in array
                # |alpha|.
        while (S.IsNotEmpty() and
               (S.contents[-1]).clean(dfsnum[parent[w]], alpha, dfsnum)):
            S.Pop()

        w = parent[w]
        #--------------------------------------------------------------


        #--------------------------------------------------------------
        # TEST STRONG PLANARITY AND COMPUTE Att
        # We test the strong planarity of the segment "S(e0)".
        # We know at this point that the interlacing graph is bipartite.
        # Also for each of its connected components the corresponding block
        # on stack |S| contains the list of attachments below |x|.
        # Let |B| be the topmost block of |S|.
        # If both sides of |B| have an attachment above |w0| then
        # "S(e0)" is not strongly planar.
        # We free the storage allocated for the blocks and return false.
        # Otherwise (cf. procedure |add_to_Att|) we first make sure that
        # the right side of |B| attaches only to |w0| (if at all) and then
        # add the two sides of |B| to the output list |Att|.
        # We also record the placements of the subsegments in |alpha|.
    Att.clear()

    while S.IsNotEmpty():
        B = S.Pop()

        if (not(B.empty_Latt()) and not(B.empty_Ratt()) and
            B.head_of_Latt() > dfsnum[w0] and B.head_of_Ratt() > dfsnum[w0]):
            B = None
            while S.IsNotEmpty(): S.Pop()
            return 0
        B.add_to_Att(Att, dfsnum[w0], alpha, dfsnum)
        B = None

        # Let's not forget that "w0" is an attachment
        # of "S(e0)" except if w0 = x.
    if w0 != x: Att.append(dfsnum[w0])

    return 1
    #--------------------------------------------------------------


    #=============================================================================#



    #=============================================================================#
def embedding(e0, t, T, A):
# embed: determine the cycle "C(e0)"
#
# We start by determining the spine cycle.
# This is precisley as in |strongly_planar|.
# We also record for the vertices w_r+1, ...,w_k, and w_0 the
# incoming cycle edge either in |tree_edge_into| or in the local
# variable |back_edge_into_w0|.

    global G, dfsnum, cur_nr, sort_num, tree_edge_into, parent

    x = source(e0)
    y = target(e0)
    tree_edge_into[y] = e0
    e = G.first_adj_edge(y)
    wk = y

    while (dfsnum[target(e)] > dfsnum[wk]):  # e is a tree edge
        wk = target(e)
        tree_edge_into[wk] = e
        e = G.first_adj_edge(wk)

    w0 = target(e)
    back_edge_into_w0 = e


    # process the subsegments
    w = wk
    Al = List()
    Ar = List()
    Tprime = List()
    Aprime = List()

    T.clear()
    T.append(e)  # |e=(wk,w0)| at this point

    while w != x:
        count = 0
        for e in G.adj_edges(w):
            count = count + 1
            if count != 1:  # no action for first edge
                # embed recursively
                if dfsnum[w] < dfsnum[target(e)]:
                    # tree edge
                    if t == alpha[e]:
                        tprime = left
                    else:
                        tprime = right
                    embedding(e, tprime, Tprime, Aprime)
                else:
                    # back edge
                    Tprime.append(e)
                    Aprime.append(reversal(e))

                    # update lists |T|, |Al|, and |Ar|
                if t == alpha[e]:
                    Tprime.conc(T)
                    T.conc(Tprime)  # T = Tprime conc T
                    Al.conc(Aprime)  # Al = Al conc Aprime
                else:
                    T.conc(Tprime)  # T = T conc Tprime
                    Aprime.conc(Ar)
                    Ar.conc(Aprime)  # Ar = Aprime conc Ar

                    # compute |w|'s adjacency list and prepare for next iteration
        T.append(reversal(tree_edge_into[w]))  # (w_j-1,w_j)
        for e in T.elements:
            sort_num[e] = cur_nr
            cur_nr = cur_nr + 1

            # |w|'s adjacency list is now computed; we clear |T| and
            # prepare for the next iteration by moving all darts incident
            # to |parent[w]| from |Al| and |Ar| to |T|.
            # These darts are at the rear end of |Al| and at the front end
            # of |Ar|.
        T.clear()

        while not(Al.empty()) and source(Al.tail()) == parent[w]:
        # |parent[w]| is in |G|, |Al.tail| in |H|
            T.push(Al.Pop())  # Pop removes from the rear

        T.append(tree_edge_into[w])  # push would be equivalent

        while not(Ar.empty()) and source(Ar.head()) == parent[w]:
            T.append(Ar.pop())  # pop removes from the front

        w = parent[w]

        # prepare the output
    A.clear()
    A.conc(Ar)
    A.append(reversal(back_edge_into_w0))
    A.conc(Al)
    #=============================================================================#


###############################################################################
###############################################################################
###############################################################################
#                                                                             #
#                            AN IMPLEMENTATION OF                             #
#                    THE "DE FRAYSSEIX,PACH,POLLACK(FPP)"                     #
#                             AND THE "SCHNYDER"                              #
#                   PLANAR STRAIGHT-LINE EMBEDDING ALGORITHM                  #
#                                                                             #
###############################################################################
#                                                                             #
# References:                                                                 #
#                                                                             #
# [FPP90] H. de Fraysseix, J.Pach, and R.Pollack .                            #
#         "How to draw a planar graph on a grid."                             #
#         Combinatorian, 10:41-51,1990                                        #
# [Sch90] W.Schnyder.                                                         #
#         "Embedding planar graphs on the grid."                              #
#         In 1st Annual ACM-SIAM Symposium on Discrete Algorithms,            #
#         pages 138-14, San Francisco, 1990                                   #
#                                                                             #
###############################################################################


class pe_Point:

    def __init__(self, xpos, ypos):
        self.x = xpos
        self.y = ypos

class pe_Node:

    def __init__(self, x, y):
        self.xpos = x
        self.ypos = y
        self.canOrder = None
        self.t1, self.t2, self.t3 = None, None, None
        self.p1, self.p2, self.p3 = None, None, None
        self.r1, self.r2, self.r3 = None, None, None
        self.xsch, self.ysch = None, None
        self.xfpp, self.yfpp = None, None
        self.adjacentEdges = []
        self.adjacentNodes = []
        self.M = []
        self.oppositeNodes = []
        self.outface = None

        self.path1 = []
        self.path2 = []
        self.path3 = []

    def addEdge(self, e, v):
        self.adjacentEdges.append(e)
        self.adjacentNodes.append(v)





class pe_Edge:  # directed from p1->p2

    def __init__(self, index_p1, index_p2, ep1, ep2, tf):
        self.p1 = index_p1
        self.p2 = index_p2
        self.label = None  # normal labelling: 1,-1,2,-2,3,-3
        self.original = tf
        self.outface = None





class pe_Graph:


    def __init__(self):
        self.nodes = []
        self.edges = []

        self.orderK, self.orderIndexVk = None, None
        self.FPPk = None
        self.labelK = None
        self.indexV1, self.indexV2, self.indexV3 = -1, -1, -1

    def checkIndex(self, index, p1):
        if index < 0:
            tempNode1 = pe_Node(p1.x, p1.y)
            self.nodes.append(tempNode1)
            return (len(self.nodes) - 1)
        return index

    def storeEdge(self, indexP1, indexP2, p1, p2, tf):
        ep1 = pe_Point(self.nodes[indexP1].xpos, self.nodes[indexP1].ypos)
        ep2 = pe_Point(self.nodes[indexP2].xpos, self.nodes[indexP2].ypos)
        self.edges.append(pe_Edge(indexP1, indexP2, ep1, ep2, tf))

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # TRIANGULATION
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Algorithm:
        # For each vertex v
        #     for v's each pair of consecutive neighbours u & w
        #             add the edge in
        #             add u into w's incident list in ccw order
        #             add w into u's incident list in ccw order
        #             repeat this procedure
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def isEdge(self, u, w):
        # check if w is in u's adjacentEdges
        if w in u.adjacentNodes: return 1
        return 0

    def adjacentVertex(self, v, e):
        return v.adjacentNodes[e]

    def consider(self):
        for indexV in range(0, len(self.nodes)):
            v = self.nodes[indexV]

            if len(v.adjacentEdges) < 2: continue

            for j in range(0, len(v.adjacentEdges)):
                # get two consective neighbours of v
                indexU = self.adjacentVertex(v, j)
                u = self.nodes[indexU]
                k = j + 1
                if k == len(v.adjacentEdges): k = 0
                indexW = self.adjacentVertex(v, k)
                w = self.nodes[indexW]

                # check if (u, w) is an edge
                if not(self.isEdge(u, indexW)):
                    pointu = pe_Point(u.xpos, u.ypos)
                    pointw = pe_Point(w.xpos, w.ypos)
                    self.storeEdge(indexU, indexW, pointu, pointw, 0)

                    tempi1 = indexV
                    tempe1 = len(self.edges) - 1

                    # add u to w's adjacentEdges (with ordering)
                    # add u after v in w's adjacentEdges
                    indexVinW = w.adjacentNodes.index(tempi1) + 1
                    w.adjacentEdges.insert(indexVinW, tempe1)
                    w.adjacentNodes.insert(indexVinW, indexU)

                    # add w to u's incitentList (with ordering)
                    # add w before v in u's adjacentEdges
                    indexVinU = u.adjacentNodes.index(tempi1)
                    u.adjacentEdges.insert(indexVinU, tempe1)
                    u.adjacentNodes.insert(indexVinU, indexW)

                    # Don't forget to set original=0
                    self.edges[-1].original = 0

                    return 1
        return 0

    def triangulate(self):
        finish = 1
        while finish:
            finish = self.consider()
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # CANONICAL ORDERING
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Algorithm:
            # Pick up a face as outface
            # Assign its vertices with canonical ordering 1,2, and n
            #
            # For k from n-1 to 3
            #     remove Vk+1 from graph
            #     find all Vk+1's neighbours in the new graph Gk
            #     update the vertices on the outface
            #     assign Vk to one of these neighbours on Ck that
            #                                              is not V1
            #                                              is not V2
            #                                              is not incident to a chord
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            #-------------------------------------------------------------------------
    def ordering(self):
        self.orderK = len(self.nodes)

        # Now, remove Vn from the graph, and let Vn-1 be the vertex
        # that is on the outerface and not incident to a chord.
        k = len(self.nodes)
        while k > 3:
            self.orderIndexVk = self.order()
            k = k - 1

    def initOrder(self):
        # NOTE: initially, all the canOrder are 0
        for i in range(0, len(self.nodes)):
            self.nodes[i].canOrder = 0

        # Base: find v1, v2, and vn, which define a outerface
        if self.indexV1 < 0:
            self.indexV1 = 0
            v1 = self.nodes[self.indexV1]

            self.indexV2 = v1.adjacentNodes[0]
            v2 = self.nodes[self.indexV2]

            self.indexVn = v1.adjacentNodes[1]
            vk = self.nodes[self.indexVn]
        else:
            v1 = self.nodes[self.indexV1]
            v2 = self.nodes[self.indexV2]
            vk = self.nodes[self.indexVn]

        v1.canOrder = 1
        v2.canOrder = 2
        vk.canOrder = len(self.nodes)

        # initialize all the outface to 0
        for j in range(0, len(self.nodes)):
            self.nodes[j].outface = 0

        self.orderK = len(self.nodes)
        self.orderIndexVk = self.indexVn
        return self.indexVn

    #-------------------------------------------------------------------------
    # Now, remove Vn from the graph, and let Vn-1 be the vertex
    # that is on the outerface and not incident to a chord
    def order(self):
        if self.orderK > 3:
            v1 = self.nodes[self.indexV1]
            v2 = self.nodes[self.indexV2]
            vk = self.nodes[self.orderIndexVk]
            # "remove" Vk from the graph
            # find the neighbours of Vk, that have canOrder number < k
            # define they are on the outface
            for i in range(0, len(vk.adjacentNodes)):
                neighbour = self.nodes[vk.adjacentNodes[i]]

                if neighbour.canOrder < self.orderK:
                    neighbour.outface = 1

                    # find the node that is not v1, v2, and not incident to any chord,
                    # let it be Vk-1
            found = 0
            j = 0
            while not(found) and j < len(self.nodes):
                candidate = self.nodes[j]
                if (candidate.outface and candidate != v1 and
                    candidate != v2 and candidate.canOrder < self.orderK):
                    # if it only has 2 neighbours on the outface,
                    # we set it to be Vk-1
                    count = 0
                    for i in range(0, len(candidate.adjacentNodes)):
                        checkIndex = candidate.adjacentNodes[i]
                        checkNode = self.nodes[checkIndex]
                        if (checkNode.outface and
                            (checkNode.canOrder < self.orderK)):
                            count = count + 1
                    if count == 2:
                        found = 1
                        vk = candidate
                        candidate.canOrder = self.orderK - 1

                        self.orderK = self.orderK - 1
                        self.orderIndexVk = j
                        return j
                j = j + 1
        return -1

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # EDGE LABELLING
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Algorithm:
        # In triangle V1, V2, V3 (canonical order)
        #     label V3->V1 with 1
        #     label V3->V2 with 2
        #
        # For k from 3 to n-1
        #     add Vk+1 to graph Gk
        #     find all Vk+1's neighbours in Gk in order
        #     label the left most edge from top to bottom with 1
        #     label the right most edge from top to bottom with 2
        #     label the rest from bottom to top with 3
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-------------------------------------------------------------------------
    def labelling(self):
        self.initLabel()
        # steps
        for k in range(3, len(self.nodes)):
            self.labelK = k
            self.labelStep()
            # looking for v4, v5, ..., vn
            #-------------------------------------------------------------------------


            #-------------------------------------------------------------------------
            # label  label if the edge is indexP1 -> indexP2
            #       -label if the edge is indexP2 -> indexP1

    def labelEdge(self, indexP1, indexP2, label):
        e = self.edges[0]

        if e.p1 == indexP1 and e.p2 == indexP2:
            self.edges[0].label = label
            return
        if e.p1 == indexP2 and e.p2 == indexP1:
            self.edges[0].label = -label
            return

        for i in range(1, len(self.edges)):
            e = self.edges[i]
            if e.p1 == indexP1 and e.p2 == indexP2:
                e.label = label
                return
            if e.p1 == indexP2 and e.p2 == indexP1:
                e.label = -label
                return


    def findIndexOfVk(self, k):
        indexVk = -1
        i = 0

        while indexVk < 0:
            if self.nodes[i].canOrder == k:
                return i
            i = i + 1
        return -1

    def initLabel(self):
        for j in range(0, len(self.edges)):
            self.edges[j].label = 0

        self.indexV3 = self.findIndexOfVk(3)

        # find v1, v2, v3
        v1 = self.nodes[self.indexV1]
        v2 = self.nodes[self.indexV2]
        v3 = self.nodes[self.indexV3]

        # labelling should be done at the same time as FPP is running
        # (because we need the outface information)
        # but we are doing this separately, for the sak of clearness

        # label V3 -> V1 by 1
        self.labelEdge(self.indexV3, self.indexV1, 1)

        # label V3 -> V2 by 2
        self.labelEdge(self.indexV3, self.indexV2, 2)

        self.labelK = 3




    def labelStep(self):
        k = self.labelK
        n = len(self.nodes)

        if k < n:
            indexVkplus1 = self.findIndexOfVk(k + 1)
            vkplus1 = self.nodes[indexVkplus1]

            # labelling should be done at the same time as FPP is running
            # (because we need the outface information)
            # case 1: vk+1 is "to the right" of vk
            # case 2: vk+1 is "to the left" of vk
            # case 3: vk+1 "covers" vk
            # all make the first element in Vk+1's oppositeNodes label 1,
            #          last                                        2,
            #             rest                                  3.
            # oppositeNodes is done in FPP

            first = vkplus1.oppositeNodes[0]
            self.labelEdge(indexVkplus1, first, 1)

            last = vkplus1.oppositeNodes[-1]
            self.labelEdge(indexVkplus1, last, 2)

            if len(vkplus1.oppositeNodes) > 2:
                for l in range(1, len(vkplus1.oppositeNodes) - 1):
                    self.labelEdge(indexVkplus1, vkplus1.oppositeNodes[l], -3)

            self.labelK = self.labelK + 1
        return self.labelK

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # FPP
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Algorithm:
        # 3. Initialize x,y coordinates and M for V1,V2, and V3
        #                               v1.M={v1,v2,v3}
        #                               v2.M={v2}
        #                               v3.M={v2,v3}
        # 4. In the canonical order, for each vertex
        #       1. find the vertices on the outface in order
        #       2. shift vertices in the subset M of Wp+1 and Wq
        #       3. calculate the x,y coordinates of Vk+1
        #       4. updating M for all the outface vertices
        #                               wi.M=wi.M+{vk+1}  for i<=p
        #                               vk+1.M=wp+1.M+{vk+1}
        #                               wj.M=wj.M  for j>=q
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def FPP(self):
        self.initFPP()

        # steps
        for k in range(3, len(self.nodes)):
            self.FPPk = k
            self.FPPstep()

    def initFPP(self):
        self.indexV3 = self.findIndexOfVk(3)

        # initialize all the outface to 0
        for j in range(0, len(self.nodes)):
            self.nodes[j].outface = 0
            self.nodes[j].xfpp = 0
            self.nodes[j].yfpp = 0
            self.nodes[j].M = []
            self.nodes[j].oppositeNodes = []

        # find v1, v2, v3
        v1 = self.nodes[self.indexV1]
        v2 = self.nodes[self.indexV2]
        v3 = self.nodes[self.indexV3]

        # basic
        v1.xfpp = 0; v1.yfpp = 0
        v2.xfpp = 2; v2.yfpp = 0
        v3.xfpp = 1; v3.yfpp = 1

        # v1.M={v1,v2,v3} v2.M={v2} v3.M={v2,v3}
        v1.M.append(self.indexV1)
        v1.M.append(self.indexV2)
        v1.M.append(self.indexV3)

        v2.M.append(self.indexV2)

        v3.M.append(self.indexV2)
        v3.M.append(self.indexV3)

        self.nodes[self.indexV1].outface = 1
        self.nodes[self.indexV2].outface = 1
        self.nodes[self.indexV3].outface = 1

        self.FPPk = 3


    def FPPstep(self):
        k = self.FPPk
        n = len(self.nodes)

        if k < n:
            indexVkplus1 = self.findIndexOfVk(k + 1)
            vkplus1 = self.nodes[indexVkplus1]

            # find the vertices on the outerface of Gk,
            # and in order of p, p+1, ..., q
            # find the neighbours of v(k+1) with CanOrder <= k,
            # and sort them according to their xfpp
            for i in range(0, len(vkplus1.adjacentNodes)):
                neighbour = self.nodes[vkplus1.adjacentNodes[i]]
                if neighbour.canOrder <= k:
                    insertPlace = -1
                    j = 0
                    while insertPlace < 0 and j < len(vkplus1.oppositeNodes):
                        if (neighbour.xfpp <
                            self.nodes[vkplus1.oppositeNodes[j]].xfpp):
                            insertPlace = j
                        j = j + 1

                    if insertPlace == -1:
                        vkplus1.oppositeNodes.append(
                            self.nodes.index(neighbour))
                    else:
                        vkplus1.oppositeNodes.insert(insertPlace,
                                                 self.nodes.index(neighbour))

                        # find the vertices on the outface
            self.nodes[indexVkplus1].outface = 1
            if len(vkplus1.oppositeNodes) > 2:
                for i in range(1, len(vkplus1.oppositeNodes) - 1):
                    temp = vkplus1.oppositeNodes[i]
                    self.nodes[temp].outface = 0

                    # shift all vertices in w(p+1).M right by 1 unit
            indexWpplus1 = vkplus1.oppositeNodes[1]
            w = self.nodes[indexWpplus1]
            for i in range(0, len(w.M)):
                temp = w.M[i]
                self.nodes[temp].xfpp = self.nodes[temp].xfpp + 1

                # shift all vertices in w(q).M right by 1 unit
            Wq = self.nodes[vkplus1.oppositeNodes[-1]]
            for i in range(0, len(Wq.M)):
                self.nodes[Wq.M[i]].xfpp = self.nodes[Wq.M[i]].xfpp + 1

                # add in v(k+1)

            Wp = self.nodes[vkplus1.oppositeNodes[0]]
            x1 = Wp.xfpp
            y1 = Wp.yfpp
            x2 = Wq.xfpp
            y2 = Wq.yfpp

            vkplus1.xfpp = (x1 + x2 + y2 - y1) / 2
            vkplus1.yfpp = (x2 - x1 + y2 + y1) / 2

            # update M
            # wi.M = wi.M + v(k+1)  for i<=p
            for i in range(0, n):
                wi = self.nodes[i]
                if wi.outface  and  wi.xfpp < w.xfpp and i != indexVkplus1:
                    wi.M.append(indexVkplus1)

                    # v(k+1).M = w(p+1).M + v(k+1)
            vkplus1.M.append(indexVkplus1)
            for i in range(0, len(w.M)):
                vkplus1.M.append(w.M[i])


            self.FPPk = self.FPPk + 1

        return self.FPPk

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SCHNYDER
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Algorithm:
    # 4. Calculate for each interior vertex v
    #              pathi : vertices on the i-path from v to v1, v2, or vn
    #              pn : number of vertices on the i-path starting at v
    #              ti : number of vertices in the subtree of Ti rooted at v
    #              ri : number of vertices in region Ri(v) for v
    # 5. Calculate barycentric representation for each v: vi'=ri - pi-1
    #                                            v->(v1',v2',v3')/(n-1)
    #    A barycentric representation of a graph G is
    #    an injective function v->(v1,v2,v3) that satisfies:
    #      a.) v1+v2+v3=1 for all v
    #      b.) for each edge (x,y) and each vertex z not x or y,
    #          there is some k (k=1,2 or 3) such that xk < zk and yk < zk
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calculateP(self, path123):
        for i in range(0, len(self.nodes)):
            v = self.nodes[i]
            finalNode = 0

            if (v.canOrder != 1 and v.canOrder != 2 and
                v.canOrder != len(self.nodes)):
                # v is an interior vertex
                if path123 == 1:
                    v.p1 = 1
                    v.path1.append(i)
                    finalNode = 1
                if path123 == 2:
                    v.p2 = 1
                    v.path2.append(i)
                    finalNode = 2
                if path123 == 3:
                    v.p3 = 1
                    v.path3.append(i)
                    finalNode = len(self.nodes)

                vNext = v
                while vNext.canOrder != finalNode:
                    j = 0
                    found = 0
                    while (not(found) and j < len(vNext.adjacentEdges)):
                        curEdge = self.edges[vNext.adjacentEdges[j]]
                        curLabel = curEdge.label
                        # HIER IST IRGENDWO EIN FEHLER !!!!!!
                        if (((curLabel == path123) and
                              (curEdge.p1 == self.nodes.index(vNext))) or
                             ((curLabel == -path123) and
                              (curEdge.p2 == self.nodes.index(vNext)))):
                            found = 1
                            vNextIndex = vNext.adjacentNodes[j]
                            vNext = self.nodes[vNextIndex]

                            if path123 == 1:
                                v.p1 = v.p1 + 1
                                v.path1.append(vNextIndex)
                            if path123 == 2:
                                v.p2 = v.p2 + 1
                                v.path2.append(vNextIndex)
                            if path123 == 3:
                                v.p3 = v.p3 + 1
                                v.path3.append(vNextIndex)
                        j = j + 1
                        #-------------------------------------------------------------------------


                        #-------------------------------------------------------------------------
    def traverse(self, label, v, count):
        for j in range(0, len(v.adjacentEdges)):
            curEdge = self.edges[v.adjacentEdges[j]]
            curLabel = curEdge.label
            if (((curLabel == -label) and
                  (curEdge.p1 == self.nodes.index(v))) or
                 ((curLabel == label) and
                  (curEdge.p2 == self.nodes.index(v)))):
                vNextIndex = v.adjacentNodes[j]
                vNext = self.nodes[vNextIndex]
                count = count + 1
                count = self.traverse(label, vNext, count)
        return count




    def Schnyder(self):
        # we need to compute p1, p2, p3 and t1, t2, t3 and
        # r1, r2, r3 for each vertex

        # Initialize the data
        for i in range(0, len(self.nodes)):
            tempnn1 = self.nodes[i]
            tempnn1.p1 = 0
            tempnn1.p2 = 0
            tempnn1.p3 = 0
            tempnn1.t1 = 0
            tempnn1.t2 = 0
            tempnn1.t3 = 0
            tempnn1.r1 = 0
            tempnn1.r2 = 0
            tempnn1.r3 = 0
            tempnn1.xsch = 0
            tempnn1.ysch = 0
            tempnn1.path1 = []
            tempnn1.path2 = []
            tempnn1.path3 = []

            # find those for v1, v2, and vn
        v1 = self.nodes[self.indexV1]
        v2 = self.nodes[self.indexV2]
        vn = self.nodes[self.indexVn]
        v1.t1 = 0; v1.t2 = 1; v1.t3 = 1;
        v2.t1 = 1; v2.t2 = 0; v2.t3 = 1;
        vn.t1 = 1; vn.t2 = 1; vn.t3 = 0;
        v1.p1 = 0; vn.p1 = 1;
        v2.p2 = 0; vn.p2 = 1;

        v1.xsch = len(self.nodes) - 1; v1.ysch = 0;
        v2.xsch = 0; v2.ysch = len(self.nodes) - 1;
        vn.xsch = 0; vn.ysch = 0;


        # can we get p1/p2/p3 while doing ordering or labelling???
        # Not really!!! We cannot get p3 easily

        # calculate p1, p2, p3 by going through path1, path2, path3
        self.calculateP(1)
        self.calculateP(2)
        self.calculateP(3)

        # calculate t1, t2, t3
        # exterior vertices are done in ordering
        # for each interior vertex v
        for i in range(0, len(self.nodes)):
            v = self.nodes[i]

            if (v.canOrder != 1 and v.canOrder != 2 and
                v.canOrder != len(self.nodes)):
                # v is an interior vertex
                v.t1 = self.traverse(1, v, 1)  # Itself is in the subtree
                v.t2 = self.traverse(2, v, 1)  # Itself is in the subtree
                v.t3 = self.traverse(3, v, 1)  # Itself is in the subtree

                # calculate r1, r2, r3
                # we need 3 vectors in each vertex to store the path 1,2,3
                # v.ri = ti of all vertices on P(i+1)+ti of all vertices on P(i-1)-ti
        for i in range(0, len(self.nodes)):
            v = self.nodes[i]

            if (v.canOrder != 1 and v.canOrder != 2 and
                v.canOrder != len(self.nodes)):
                # v is an interior vertex
                v.r1 = 0; v.r2 = 0; v.r3 = 0;

                # r1
                for j in range(0, len(v.path2)):
                    onPath = self.nodes[v.path2[j]]
                    v.r1 = v.r1 + onPath.t1
                for j in range(0, len(v.path3)):
                    onPath = self.nodes[v.path3[j]]
                    v.r1 = v.r1 + onPath.t1
                v.r1 = v.r1 - v.t1

                # r2
                for j in range(0, len(v.path1)):
                    onPath = self.nodes[v.path1[j]]
                    v.r2 = v.r2 + onPath.t2
                for j in range(0, len(v.path3)):
                    onPath = self.nodes[v.path3[j]]
                    v.r2 = v.r2 + onPath.t2
                v.r2 = v.r2 - v.t2

                # r3
                for j in range(0, len(v.path1)):
                    onPath = self.nodes[v.path1[j]]
                    v.r3 = v.r3 + onPath.t3
                for j in range(0, len(v.path2)):
                    onPath = self.nodes[v.path2[j]]
                    v.r3 = v.r3 + onPath.t3
                v.r3 = v.r3 - v.t3


                # The coordinates of each vertex is (v'1, v'2),
                # where v'i = ri - p(i-1)
                # exterior vertices are done in ordering
                # for each interior vertex v
        for i in range(0, len(self.nodes)):
            v = self.nodes[i]

            if (v.canOrder != 1 and v.canOrder != 2 and
                v.canOrder != len(self.nodes)):
                # v is an interior vertex
                v.xsch = (v.r1 - v.p3)
                v.ysch = (v.r2 - v.p1)


#=============================================================================#
# LOAD GRAPH
#=============================================================================#
def load_graph(InGraph):

    if InGraph.order() < 3:
        return 0

    ccwOrderedEdges = planarity_test(InGraph)
    if not(ccwOrderedEdges):
        # raise RuntimeError("Planarity Test", "Graph is NOT PLANAR!")
        return 0

    graph1 = pe_Graph()

    i = 0
    NodeIndex = {}
    for v in sorted(InGraph.vertices):
        NodeIndex[v] = i
        i = i + 1

    for i in range(0, InGraph.order()):
        graph1.nodes.append(pe_Node(i, i))

    EdgeIndex = {}
    for i in range(0, len(ccwOrderedEdges)):
        n1 = NodeIndex[ccwOrderedEdges[i][0]]
        n2 = NodeIndex[ccwOrderedEdges[i][1]]
        ccwOrderedEdges[i] = (n1, n2)
        EdgeIndex[(n1, n2)] = None

    i = 0
    for e in ccwOrderedEdges:
        if EdgeIndex[e] == None:
            EdgeIndex[e] = i
            EdgeIndex[(e[1], e[0])] = i
            i = i + 1
            p1 = pe_Point(e[0], e[0])
            p2 = pe_Point(e[1], e[1])
            tempe1 = pe_Edge(e[0], e[1], p1, p2, 1)
            graph1.edges.append(tempe1)

        graph1.nodes[e[0]].addEdge(EdgeIndex[e], e[1])

    return graph1
#=============================================================================#



class Rect:
    def __init__(self):
        self.x1 = 30
        self.y1 = 30
        self.x2 = 600
        self.y2 = 400

    def midpoint(self):
        return ((self.x2 - self.x1) / 2.0, (self.y2 - self.y1) / 2.0)

    def height(self):
        return self.x2 - self.x1

    def width(self):
        return self.y2 - self.y1

def FPP_planar_coords(G, rect = None):  # (2n-4)*(n-2) GRID
# Algorithm:
# 1. Triangulate orginal graph
# 2. Canonical order all vertices
# 3. Initialize x,y coordinates and M for V1,V2, and V3
#                               v1.M={v1,v2,v3}
#                               v2.M={v2}
#                               v3.M={v2,v3}
# 4. In the canonical order, for each vertex
#       1. find the vertices on the outface in order
#       2. shift vertices in the subset M of Wp+1 and Wq
#       3. calculate the x,y coordinates of Vk+1
#       4. updating M for all the outface vertices
#                               wi.M=wi.M+{vk+1}  for i<=p
#                               vk+1.M=wp+1.M+{vk+1}
#                               wj.M=wj.M  for j>=q

    #-------------------------------------------------------------------------
    # LOAD GRAPH
    graph = load_graph(G)
    if graph == 0: return False
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    # 1.TRIANGULATION
    graph.triangulate()
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    # 2.CANONICAL ORDERING
    graph.initOrder()
    graph.ordering()
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    # 3+4. FPP
    graph.FPP()
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    if rect is None:
        rect = Rect()
    # COORDINATES
    pos = {}
    n = len(graph.nodes)
    for i in range(0, n):
        xCoord = graph.nodes[i].xfpp * float((rect.height() - 100) / (2 * n - 4)) + 50
        yCoord = rect.width() - (graph.nodes[i].yfpp * float((rect.width() - 100) / (n - 2)) + 50)
        pos[sorted(G.vertices)[i]] = (xCoord + random.random(), yCoord + random.random())

    return pos


def Schnyder_planar_coords(G, rect = None):  # (n-1)*(n-1) GRID
# Algorithm:
# 1. Triangulate orginal graph
# 2. Canonical order all vertices
# 3. Normal label interior edges of G to i->Ti (i=1,2,3)
# 4. Calculate for each interior vertex v
#              pathi : vertices on the i-path from v to v1, v2, or vn
#              pn : number of vertices on the i-path starting at v
#              ti : number of vertices in the subtree of Ti rooted at v
#              ri : number of vertices in region Ri(v) for v
# 5. Calculate barycentric representation for each v: vi'=ri - pi-1
#                                            v->(v1',v2',v3')/(n-1)
#    A barycentric representation of a graph G is
#    an injective function v->(v1,v2,v3) that satisfies:
#      a.) v1+v2+v3=1 for all v
#      b.) for each edge (x,y) and each vertex z not x or y,
#          there is some k (k=1,2 or 3) such that xk < zk and yk < zk

    #-------------------------------------------------------------------------
    # LOAD GRAPH
    graph = load_graph(G)
    if graph == 0: return False
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    # 1.TRIANGULATION
    graph.triangulate()
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    # 2.CANONICAL ORDERING
    graph.initOrder()
    graph.ordering()
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    # 3. EDGE LABELLING
    graph.FPP()  # outfaces
    graph.labelling()
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    # 4+5. Schnyder
    graph.Schnyder()
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    if rect is None:
        rect = Rect()

    # COORDINATES
    pos = {}
    n = len(graph.nodes)
    for i in range(0, n):
        xCoord = graph.nodes[i].xsch * float((rect.height() - 100) / (n - 1)) + 50
        yCoord = rect.width() - (graph.nodes[i].ysch * float((rect.width() - 100) / (n - 1)) + 50)
        pos[sorted(G.vertices)[i]] = (xCoord + random.random(), yCoord + random.random())
    return pos




