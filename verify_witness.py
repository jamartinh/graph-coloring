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

def verify():
    graph_file = sys.argv[1]
    witness_file = sys.argv[2]
    main(graph_file, witness_file)

if __name__ == '__main__':
    verify()
