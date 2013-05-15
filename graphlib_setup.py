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
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
from itertools import ifilter

module1 = Extension('graphlib',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['./planarity'],
                    sources = ['graphlib.pyx']
                    )

module1.extra_compile_args = ['-O3']

setup(
  name = "3-coloring library",
  cmdclass = {'build_ext': build_ext},
  ext_modules = [module1]
)

