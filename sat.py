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
import operator
import random



__doc__ = """
This is a small boolean satisfiability library (3-SAT).

A 3-SAT instance is represented as a list of clauses.
A clause is represented as a list of integers from 1 to N
where 1 means the variable x_1 and -1 means the negation of variable x_1,
e.g. [-1,2,-4] is (not x_1, x_2, not x_4)

to copy an instance use map(list,instance) to obtain a fast deepcopy

"""

def random_3sat_clause(n_vars):
    """
    returns a random 3-sat clause for instances of n_vars
    """
    signs = [1, 1, 1, -1, -1, -1]
    x = random.sample(range(1, n_vars + 1), 3)
    s = random.sample(signs, 3)
    return sorted(map(operator.mul, x, s), key = abs)


def random_3sat_instance(n_vars, m_clauses):
    """
    returns a random 3-sat instance of n_vars variables and m_clauses clauses
    """

    instance = list()
    while(len(instance) < m_clauses):
        clause = random_3sat_clause(n_vars)
        # if clause not in instance:
        instance.append(clause)

    return instance

def verify_3sat_solution(instance, solution):
    """ verify solution over instance
        solution should be a sequence of 0/1 or True/False elements indicating
        if variable ith is 1 or 0
        -1 means that the value is undefined indicating a partial solution,
        so you can test if a partial solution violates a clause. This is
        useful, for instance, for backtracking heuristics
    """

    for a, b, c in instance:
        va, vb, vc = solution[abs(a) - 1], solution[abs(b) - 1], solution[abs(c) - 1]
        if (va == -1) or (va and a > 0) or (not va and a < 0): continue
        if (vb == -1) or (vb and b > 0) or (not vb and b < 0): continue
        if (vc == -1) or (vc and c > 0) or (not vc and c < 0): continue
        return False

    return True


def ratio_3sat(N, M):
    return M / float(N)

def clauses_from_ratio(N, alpha = 4.25):
    return alpha * N


#-------------------------------------------------------------------------------
# Backtracking 3-SAT algorithm
#-------------------------------------------------------------------------------

def bt_3sat(n_vars, i, solution, instance):
    """
    2^N brute force algorithm for 3-sat  with backtracking.
    """
    if i >= n_vars: return 1, solution

    solution[i] = 0
    if verify_3sat_solution(instance, solution):
        Q, solution_r = bt_3sat(n_vars, i + 1, list(solution), instance)
        if Q: return 1, solution_r

    solution[i] = 1
    if verify_3sat_solution(instance, solution):
        Q, solution_r = bt_3sat(n_vars, i + 1, list(solution), instance)
        if Q: return 1, solution_r



    return 0, None

def is_satisfiable_3sat_BF(instance):
    """
    brute force backtracking algorithm
    """
    min_var = min(map(min, instance))
    max_var = max(map(max, instance))
    variables = sorted(set([abs(v) for clause in instance for v in clause]))
    # n_vars = (max_var - min_var)
    n_vars = len(variables)
    solution = [-1] * n_vars

    Q, solution = bt_3sat(n_vars, 0, list(solution), instance)
    return Q, solution

def bt_3sat_witness(solution):
    C = dict()
    for i, value in enumerate(solution):
        C[i + 1] = value

    str_sol = repr(C)
    replace_tuples = ((',', ',\n'), (']', '\n]'), ('[', '\n[\n '), ('{', '{\n'), ('}', '\n}'))
    for older, newer in replace_tuples: str_sol = str_sol.replace(older, newer)
    return str_sol



def load_from_cnf(strFileName = None, str_text = None):

    instance = list()
    if strFileName is not None:
        with open(strFileName, 'r') as f:
            lines = f.readlines()
    else:
        lines = str_text.splitlines()


    while len(lines[0]) == 0 or lines[0][0] == 'c' or lines[0][0] == 'p':
        lines.pop(0)
        if lines[0][0] == 'p' and 'cnf' not in lines[0]: raise TypeError

    str_text = " ".join(lines)
    lines = str_text.split('0')
    lines.pop()

    for line in lines:
        fields = line.split()
        if len(fields) < 3: continue
        if line[0] == '%': continue
        instance.append((int(fields[0]), int(fields[1]), int(fields[2])))

    return instance




def main():
    import time
    yes = 0
    no = 0
    TOP = 100
    for _i in range(TOP):
        N = 30
        instance = random_3sat_instance(N, clauses_from_ratio(N, 4.5))
        # print instance
        t1 = time.clock()
        Q, _s = is_satisfiable_3sat_BF(instance)
        t2 = time.clock()
        if Q:
            yes += 1
        else:
            no += 1
        print "solucion", Q, t2 - t1

        # print s
        # G = reduce_3sat_to_3col(instance)
        # from Gato.Gred import DrawMyGraph
        # print G.vertices
        # print G.edges
        # DrawMyGraph(G)
    print "totals", yes, no, (yes / float(TOP)) * 100, "%"



if __name__ == '__main__':
    main()
