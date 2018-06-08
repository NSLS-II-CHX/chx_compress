#!/usr/bin/env python
import setuptools
from distutils.core import setup, Extension
import versioneer
import os
import sys
#from Cython.Build import cythonize

# necessary deps (won't build if this isn't here)
import numpy as np


with open('requirements.txt') as f:
    requirements = f.read().split()

setuptools.setup(
    name='chx_xpcs',
    author='Yugang Zhang and Julien Lhermitte',
    packages=setuptools.find_packages(exclude=['doc']),
    include_dirs=[np.get_include()],
    install_requires=requirements
    # scripts for building external modules
    #ext_modules=cython_ext(),
    )

