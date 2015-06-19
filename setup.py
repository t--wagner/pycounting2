# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = 'cycounting2',
    ext_modules = cythonize("cycounting2.pyx"),
    #include_dirs = [np.get_include()]
)