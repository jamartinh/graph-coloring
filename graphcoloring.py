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
# import pyximport
# pyximport.install(pyimport = True)

import graphio as gio
import reduce3col as r3
import sys
import testutil as tu
import time

#-----------------------------------------------------------------------------------------------

def main(strFileName, alpha = 5):
    print " *** Starting *** "
    G = gio.load_from_edge_list_named(strFileName)

    V, E, avgdeg = G.order(), G.size(), G.avg_degree()
    t1 = time.clock()
    Q, G, P, max_alpha = r3.incremental_depth_3COL(G, alpha)
    t2 = time.clock()
    strScreen = "3-COL: %d N: %d M: %d time1: %3.3f average degree: %2.2f alpha: %s" % (bool(Q), V, E, t2 - t1, avgdeg, max_alpha)
    print strScreen
    if not Q:
        witness = r3.UNCOL_witness(P)
    else:
        witness = r3.COL_witness(G, P)

    tu.save_witness("", Q, witness, "", strFileName)


def main2(strFileName, alpha = 5):
    print " *** Starting *** "
    G = gio.load_from_edge_list_named(strFileName)

    V, E, avgdeg = G.order(), G.size(), G.avg_degree()
    t1 = time.clock()
    Q, G, P = r3.general_3COL(G, alpha)
    t2 = time.clock()
    strScreen = "3-COL: %d N: %d M: %d time1: %3.3f average degree: %2.2f alpha: %s" % (Q, V, E, t2 - t1, avgdeg, alpha)
    print strScreen
    if not Q:
        witness = r3.UNCOL_witness(P)
    elif Q == 1:
        witness = r3.COL_witness(G, P)
    else:
        return

    tu.save_witness("", Q, witness, "", strFileName)

if __name__ == "__main__":
    if len(sys.argv) == 3: main(sys.argv[1], int(sys.argv[2]))
    if len(sys.argv) == 4: main2(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 2: main(sys.argv[1])
# main("kinstances/IF.col", 5)
# main("to_3coloring_uf50-0955.cnf.col", 5)
# main("to_3coloring_IF.col", 5)
# main("kinstances/planar_IF.col", 2)
# main("kinstances/1-insertions_4.col", 2)
