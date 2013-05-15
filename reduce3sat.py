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
from collections import defaultdict, Counter
from itertools import combinations, izip, product, groupby
from sat import *
import math as m
import operator as op
import random




opcodes = ['K8', 'A', 'C', 'U']
UNDETERMINED = 3
def z(s):
    return frozenset(z)

def decode_operation(O, prefix = '', recursive_level = 1):


    if prefix != '': prefix = '  ' * recursive_level + prefix + ' '
    op = opcodes[O[0]]
    if   op == 'K8':    return prefix + 'Q.E.D. K8 ' + str(O[1])
    elif op == 'K7':    return prefix + 'K7 ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + ' ' + str(sorted(O[3])) + '\n'
    elif op == 'U':     return prefix + 'U ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + ' ' + str(sorted(O[3])) + '\n'
    elif op == 'R':     return prefix + 'R ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + ' ' + str(sorted(O[3])) + '\n'
    elif op == 'K4':    return prefix + 'K41 ' + str(sorted(O[1])) + ' ' + str(sorted(O[2])) + '\n' + '  ' * recursive_level + 'BEGIN\n' + UNSAT_witness(O[3], 'SUBT31', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'
    elif op == 'C':     return prefix + 'C ' + str(sorted(O[1])) + '\n' + '  ' * recursive_level + 'BEGIN\n' + UNSAT_witness(O[3], 'SUBA', recursive_level + 1) + '\n' + '  ' * recursive_level + 'END\n'




def UNSAT_witness(P, prefix = '', recursive_level = 1):
    pf_text = ''
    values = sorted(P.values(), key = lambda x: x[-1])
    for o in values:
        pf_text += decode_operation(o, prefix, recursive_level)

    return pf_text

def SAT_witness(G, P = None):

    if P: return UNSAT_witness(P)


    C = dict()
    for color, v in enumerate(G.identities.iterkeys(), 1):
        C[color] = sorted(G.identities[v])

    str_col = repr(C)
    replace_tuples = ((',', ',\n'), (']', '\n]'), ('[', '\n[\n '), ('{', '{\n'), ('}', '\n}'))
    for older, newer in replace_tuples: str_col = str_col.replace(older, newer)

    return str_col


################################################################################
#  Much faster methods than deepcopy
################################################################################

def dict_copy(source):
    """ copy dict of sets"""
    # return dict((k, set(v)) for (k, v) in source.iteritems())
    return dict(zip(source.keys(), map(set, source.values())))
    # return {k:v for k, v in zip(source.keys(), map(set, source.values()))}


def list_copy(source):
    """ copy list of iterable to list of lists"""
    return map(list, source)

def set_copy(source):
    """copy set of sets"""
    return set(map(frozenset, source))


# class TreeCounter(dict):



class sat_instance:

    def __init__(self, nvars = 0):
        self.clauses = set()
        self.variables = set()
        self.count = Counter()
        self.identities = defaultdict(set)  # a dictionary of identified vertices
        self.domain_count = dict()


    def __iter__(self):
        """Iterates over the clauses.
        """
        return iter(self.clauses)

    def __contains__(self, clause):
        """Return True if c is a clause.
        (a,b,c) in F ?
        [a,b,c] in F ?
        {a,b,c} in F ?
        """
        return set(clause) in self.clasues

    def __len__(self):
        # return len(self.clauses)
        return len(self.variables)

    def __getitem__(self, lt):
        """Return a set of neighbors of literal v,  'F[v]'
        """
        return {c for c in self.clauses if lt <= c}

    def degree(self, lt):
        lt = set(lt)
        return len(filter(lt.intersection, self.clauses))

    def clear(self, first_v = None):
        self.__init__()

    def copy(self):
        F = self.__class__()
        # copy the minimum set of properties to get an operative graph copy
        F.clauses = set_copy(self.clauses)
        F.variables = set(self.variables)
        F.count = Counter(self.count)
        F.identities = dict_copy(self.identities)

        # if not enough, you can always use deepcopy after all!

        return F
    def add_clause(self, new_clause):
        self.clauses.add(frozenset(new_clause))
        self.count[frozenset({abs(lt) for lt in new_clause})] += 1

    def __add__(self, new_clause):
        F = self.copy()
        F.add_clause(new_clause)
        return F

    def __iadd__(self, new_clause):
        self.add_clause(new_clause)

    def remove_clause(self, old_clause):
        self.clauses.discard(old_clause)
        self.count[frozenset({abs(lt) for lt in old_clause})] -= 1

    def __sub__(self, old_clause):
        F = self.copy()
        F.remove_clause(old_clause)
        return F

    def __isub__(self, old_clause):
        self.remove_clause(old_clause)

    def contract(self, lts):
        a = 0
        la = min(map(a.__sub__, lts))
        lts.discard(-1 * la)
        for c in set(self.clauses):
            for lb in set(lts):
                if lb in c:
                    self.remove_clause(c)
                    self.add_clause(frozenset((c - {lb}) | {la}))

                self.variables.discard(abs(lb))
                self.identities[abs(la)] = lb * m.copysign(1, la)

    def assign(self, variables, values):
        for c in set(self.clauses):
            for variable, value in izip(variables, values):
                if (-variable in c and value < 0) or (variable in c and value > 0):
                    self.remove_clause(c)
                    break
                elif (-variable in c and value > 0) or (variable in c and value < 0):
                    self.remove_clause(c)
                    c -= {variable, -variable}
                    self.add_clause(frozenset(c))



        self.variables.discard(variable)
        self.identities.update(zip(variables, values))



    def __div__(self, lts):
        F = self.copy()
        F.contract(lts)
        return F

    def __truediv__(self, lts):
        F = self.copy()
        F.contract(lts)
        return F

    def  __idiv__(self, lts):
        self.contract(lts)

    def  __itrueidiv__(self, lts):
        self.contract(lts)

    def literals(self):
        lts = set()
        map(lts.update, self.clauses)
        return lts

    def most_commom(self, n = None):
        return self.count.most_common(n)



def unit_propagation(F, P):
    for clause in sorted(F.clauses):
        if clause not in F.clauses: continue
        if len(clause) == 1:
            a = clause.pop()
            F.assign(abs(a), m.copysign(1, a))
            P[a] = (1, (abs(a), m.copysign(1, a)), len(P))

    return F, P



def check_pair(F, a, b, P):
    """
    Check all clauses of F that contain a and b
    """

    # values = Counter({{a, b}: 2, {a, -b}: 2, {-a, b}: 2, {-a, -b}:2})
    values = set_copy(product((-a, a), (-b, b)))
    idents = dict(set)
    ab = frozenset({a, b})
    idents[ab] = {True, False}

    group_key = lambda c: abs((c - {-a, a, -b, b}).pop()) if len(c - {-a, a, -b, b}) else 0
    clauses = sorted(F.clauses, key = group_key)
    for c, g in groupby(clauses, key = group_key):
        ac = frozenset({a, c})
        bc = frozenset({b, c})
        idents[ac] = {True, False}
        idents[bc] = {True, False}
        v_values = Counter({a: 4, -a: 4, b: 4, -b:4, -c:4, c:4})
        # c_values = Counter({frozenset({a, b}): 2, frozenset({a, -b}): 2, frozenset({-a, b}): 2, frozenset({-a, -b}):2})
        c_values = Counter({frozenset(s):2 for s in product((-a, a), (-b, b))})
        # c_values = set_copy(product((-a, a), (-b, b)))

        # v_values -= set(g)
        # reduce(set().union, (map(frozenset, combinations(c, 2)) for c in g))
        for la, lb, lc in g:
            # assigments to 1 or -1
            v_values -= {-la, -lb, -lc}
            c_values -= {-la, -lb}

            # assignments between vars
            idents[ab].intersection_update(m.copysign(1, la) == m.copysign(1, lb))
            idents[ac].intersection_update(m.copysign(1, la) == m.copysign(1, lc))
            idents[bc].intersection_update(m.copysign(1, lb) == m.copysign(1, lc))

        # assignments
        for v, n in v_values.most_common().reverse():
            if n < 1:
                P['QED'] = (0, (a, b, c), len(P))
                return 0, F, P
            elif n == 1:
                F.assign(abs(v), m.copysign(1, v))
                P[v] = (1, (a, b, c), len(P))
            else: break

        # identities
        for x, y in ({a, b}, {a, c}, {b, c}):
            if {x, y} <= F.variables and len(idents[frozenset({x, y})]) == 1 and x != 0 and y != 0:
                if idents[frozenset({x, y})]:
                    F.contract({x, y})
                    P[v] = (1, (a, b, c), len(P))
                else:
                    F.contract({x, -y})
                    P[v] = (1, (a, b, c), len(P))

        values &= set(c_values.elements())
        if len(values) == 0:
            P['QED'] = (0, (a, b), len(P))
            return 0, F, P

    # assignments
    if len(values) == 1:
        la, lb = values.pop()
        F.assign((abs(la), abs(lb)), (m.copysign(1, la), m.copysign(1, lb)))
        P[v] = (1, (a, b), len(P))
    # if {-a,-b} not in values:


    for v, n in c_values.most_common().reverse():
        if n < 1:
            P['QED'] = (0, (a, b), len(P))
            return 0, F, P
        elif n == 1:
            F.assign(abs(v), m.copysign(1, v))
            P[v] = (1, (a, b), len(P))
        else: break


    return UNDETERMINED, F, P



def simplify(F, P):
    F, P = unit_propagation(F, P)
    N = len(F) + 1
    while len(F) < N:
        N = len(F)
        for c, _count in  F.most_common(1):
            for a, b in combinations(c, 2):
                if {a, b} <= F:
                    Q, F, P = check_pair(F, a, b, P)
                    if Q == 0: return Q, F, P


    return UNDETERMINED


def step_into(F, P, alpha):

    for a, b in combinations(F.variables, 2):
        FF = F.copy()
        FF.contract({a, b})
        Q, FF, Pr = is_satisfiable(FF, alpha)
        if not Q:
            P[b] = [opcodes.index('C'), (a, b), None, Pr, len(P)]
            F.contract((a, -b))
            return F, P

    return F, P

def is_satisfiable(F, alpha = 0):


    P = dict()
    N = len(F) + 1

    while len(F) < N:
        N = len(F)

        # simplyfy F by applygin a set of low polynomial rules.
        Q, F, P = simplify(F, P)
        if not Q:       return 0, F, P
        if len(F) <= 2: return 1, F, P

        # step into rule.
        if alpha:
            F, P = step_into(F, P, alpha - 1)


    return UNDETERMINED, F, P

def find_non_clause(F, P):
    """
    Find a non-existent 3-clause sequentially.
    Select a clause, invert the sign of one literal
    If the clause exists, apply resolution.
    Repeat until obtaining a non-resoluble non-existent clause.  
    """
    for clause in F.clauses:
        if len(clause) < 3: continue
        a, b, c = clause
        if {a, b, -c} not in F.clauses:
            return {a, b, -c}, F, P
        else:
            P[b] = [opcodes.index('C'), (a, b), None, P, len(P)]
            F.contract({a, -b})


    return None, F, P



def find_model(F, alpha = 1):

    Q, F, P = is_satisfiable(F, alpha)
    if Q in (0, 1): return Q, F, P
    F_out = F.copy()

    # cases = tuple(product([-1, 1], repeat = 3))
    while len(F) > 2:
        # c, _count = F.most_common(1)[0]
        # _values, nonclauses = build_3group(F, c)
        # x, y, z = nonclauses.pop()
        nonclause, F, P = find_non_clause(F, P)
        x, y, z = nonclause




        # test contraction G/x,y
        H = F.copy()
        Q, F, P = is_satisfiable(F / {x, y, z}, alpha)
        if Q == 1: return 1, F, P  # model found
        # contraction failed, hence add clause x,y,z
        if not Q:
            F = H.copy()
            F.add_clause({x, y, z})

        if len(F) <= 2: return 1, F, dict()





    return UNDETERMINED, F_out, dict()




