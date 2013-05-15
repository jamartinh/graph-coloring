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

# Dijkstra's algorithm for shortest paths
# David Eppstein, UC Irvine, 4 April 2002

# Priority dictionary using binary heaps
# David Eppstein, UC Irvine, 8 Mar 2002

# The original algorithm has been modified by Jose Antonio Martin H.
# to incorporate the posibility of giving a list of possible end vertices
# so that the algorithm can be used to find the shortest path from start to
# the nearest valid possible end-point and giving its distance.

from collections import deque, defaultdict

def bfs(G, start):
    queue, enqueued = deque([(None, start)]), set([start])
    while queue:
        parent, n = queue.popleft()
        yield parent, n
        new = set(G[n]) - enqueued
        enqueued |= new
        queue.extend([(n, child) for child in new])

def dfs(G, start):
    stack, enqueued = [(None, start)], set([start])
    while stack:
        parent, n = stack.pop()
        yield parent, n
        new = set(G[n]) - enqueued
        enqueued |= new
        stack.extend([(n, child) for child in new])

def shortest_path(G, start, end):
    if isinstance(end, int): end = set([end])
    else: end = set(end)
    visited = set()

    paths = defaultdict(list)
    for parent, child in bfs(G, start):
        paths[child] = paths[parent] + [child]
        visited.add(child)
        if end <= visited:
            best_end = min(end, key = lambda x: len(paths[x]))
            return paths[best_end]
    return None  # or raise appropriate exception


class priorityDictionary(dict):
    def __init__(self):
        '''Initialize priorityDictionary by creating binary heap
        of pairs (value,key).  Note that changing or removing a dict entry will
        not remove the old pair from the heap until it is found by smallest() or
        until the heap is rebuilt.'''
        self.__heap = []
        dict.__init__(self)

    def smallest(self):
        '''Find smallest item after removing deleted items from heap.'''
        if len(self) == 0:
            raise IndexError, "smallest of empty priorityDictionary"
        heap = self.__heap
        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            lastItem = heap.pop()
            insertionPoint = 0
            while 1:
                smallChild = 2 * insertionPoint + 1
                if smallChild + 1 < len(heap) and \
                        heap[smallChild] > heap[smallChild + 1]:
                    smallChild += 1
                if smallChild >= len(heap) or lastItem <= heap[smallChild]:
                    heap[insertionPoint] = lastItem
                    break
                heap[insertionPoint] = heap[smallChild]
                insertionPoint = smallChild
        return heap[0][1]

    def __iter__(self):
        '''Create destructive sorted iterator of priorityDictionary.'''
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iterfn()

    def __setitem__(self, key, val):
        '''Change value stored in dictionary and add corresponding
        pair to heap.  Rebuilds the heap if the number of deleted items grows
        too large, to avoid memory leakage.'''
        dict.__setitem__(self, key, val)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v, k) for k, v in self.iteritems()]
            self.__heap.sort()  # builtin sort likely faster than O(n) heapify
        else:
            newPair = (val, key)
            insertionPoint = len(heap)
            heap.append(None)
            while insertionPoint > 0 and \
                    newPair < heap[(insertionPoint - 1) // 2]:
                heap[insertionPoint] = heap[(insertionPoint - 1) // 2]
                insertionPoint = (insertionPoint - 1) // 2
            heap[insertionPoint] = newPair

    def setdefault(self, key, val):
        '''Reimplement setdefault to call our customized __setitem__.'''
        if key not in self:
            self[key] = val
        return self[key]




def Dijkstra(G, start, end = None):
    """
    Find shortest paths from the start vertex to all
    vertices nearer than or equal to the end.

    The input graph G is assumed to have the following
    representation: A vertex can be any object that can
    be used as an index into a dictionary.  G is a
    dictionary, indexed by vertices.  For any vertex v,
    G[v] is itself a dictionary, indexed by the neighbors
    of v.  For any edge v->w, G[v][w] is the length of
    the edge.  This is related to the representation in
    <http://www.python.org/doc/essays/graphs.html>
    where Guido van Rossum suggests representing graphs
    as dictionaries mapping vertices to lists of neighbors,
    however dictionaries of edges have many advantages
    over lists: they can store extra information (here,
    the lengths), they support fast existence tests,
    and they allow easy modification of the graph by edge
    insertion and removal.  Such modifications are not
    needed here but are important in other graph algorithms.
    Since dictionaries obey iterator protocol, a graph
    represented as described here could be handed without
    modification to an algorithm using Guido's representation.

    Of course, G and G[v] need not be Python dict objects;
    they can be any other object that obeys dict protocol,
    for instance a wrapper in which vertices are URLs
    and a call to G[v] loads the web page and finds its links.

    The output is a pair (D,P) where D[v] is the distance
    from start to v and P[v] is the predecessor of v along
    the shortest path from s to v.

    Dijkstra's algorithm is only guaranteed to work correctly
    when all edge lengths are positive. This code does not
    verify this property for all edges (only the edges seen
     before the end vertex is reached), but will correctly
    compute shortest paths even for some graphs with negative
    edges, and will raise an exception if it discovers that
    a negative edge has caused it to make a mistake.
    """

    D = {}  # dictionary of final distances
    P = {}  # dictionary of predecessors
    Q = priorityDictionary()  # est.dist. of non-final vert.
    Q[start] = 0

    for v in Q:
        D[v] = Q[v]
        # is it valid to return when the first end-point is found?
        # break if all end-point vertices have been visited end is subset of keys
        if end.issubset(D.keys()): break
        # if v in end: break

        for w in G[v]:
            vwLength = D[v] + 1  # G[v][w]
            if w in D:
                if vwLength < D[w]:
                    raise ValueError, \
  "Dijkstra: found better path to already-final vertex"
            elif w not in Q or vwLength < Q[w]:
                Q[w] = vwLength
                P[w] = v

    return D, P

def weighted_shortest_path(G, start, end):
    """
    Find a single shortest path from the given start vertex
    to the nearest of the given set of end vertices.
    The input has the same conventions as Dijkstra().
    The output is a list of the vertices in order along
    the shortest path.
    """
    if isinstance(end, int):
        end = set([end])
    else:
        end = set(end)

    D, P = Dijkstra(G, start, end)

    # subD = {v: D.get(v,len(D)+10) for v in end} # extract only the end point values
    subD = dict((v, D[v]) for v in end)  # extract only the end point values

    best_end = min(subD, key = subD.get)  # key with min value in dict:D

    Path = [best_end]
    while best_end != start:
        best_end = P[best_end]
        Path.append(best_end)
    Path.reverse()
    return Path

