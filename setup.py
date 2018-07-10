# -*- coding: utf-8 -*-
#
# Licensed under the BSD-3-Clause license
# Copyright (c) 2018, Andres A. Perez Hortal
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from setuptools import setup, find_packages
from setuptools.extension import Extension


__author__ = "Andres Perez Hortal"
__copyright__ = "Copyright (c) 2018, Andres A. Perez Hortal, McGill University"
__license__ = "BSD-3-Clause License, see LICENCE.txt for more details"
__email__ = "andresperezcba@gmail.com"


try:
    import numpy
except ImportError:
    raise RuntimeError(
        "Numpy required to pior running the package installation\n" +
        "Try installing it with:\n" +
        "$> pip install numpy")


try:
    from Cython.Build.Dependencies import cythonize
    CYTHON_PRESENT = True
except ImportError:
    CYTHON_PRESENT = False


_vet_extension_arguments = dict(extra_compile_args=['-fopenmp'],
                                include_dirs=[numpy.get_include()],
                                language='c',
                                extra_link_args=['-fopenmp']
                                )

if CYTHON_PRESENT:
    _vet_lib_extension = Extension(str("pyVET._vet"),
                                   sources=[str('pyVET/_vet.pyx')],
                                   **_vet_extension_arguments)

    external_modules = cythonize([_vet_lib_extension])
else:
    _vet_lib_extension = Extension(str(str("pyVET._vet")),
                                   sources=[str('pyVET/_vet.c')],
                                   **_vet_extension_arguments)
    external_modules = [_vet_lib_extension]


build_requires = ['numpy']


setup(
    name='pyVET',
    version='1.0',
    author="Andres Perez Hortal",
    author_email="andresperezcba@gmail.com",
    packages=find_packages(),
    ext_modules=external_modules,
    # url='http://pypi.python.org/pypi/pyVET/',
    license='LICENSE.txt',
    description='Variational Echo Tracking Algorithm',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable', 'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Cython'],
    setup_requires=build_requires,
    install_requires=build_requires
)
