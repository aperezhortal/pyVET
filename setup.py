""" 
Setup
"""
from __future__ import division, absolute_import, print_function


__author__ = "Andres Perez Hortal"
__copyright__ = "Copyright (c) 2017, Andres A. Perez Hortal, McGill University"
__license__ = "BSD-3-Clause License, see LICENCE.txt for more details"
__email__ = "andresperezcba@gmail.com"





from setuptools import setup, find_packages
from setuptools.extension import Extension

try:
    import numpy
except ImportError:
    raise RuntimeError( "Numpy required to pior running the package installation\n" +
                        "Try installing it with:\n" + 
                        "$> pip install numpy" )
    
    
try:
    from Cython.Build.Dependencies import cythonize
    CythonPresent = True
except ImportError:
    CythonPresent = False
    

_VET_ExtensionArguments = dict(extra_compile_args = ['-fopenmp'],
                               include_dirs=[numpy.get_include()],
                               language='c',
                               extra_link_args = ['-fopenmp'] 
                               )

if CythonPresent:
    _VETLibExtension = Extension( "pyVET._VET", 
                                  sources = ['pyVET/_VET.pyx'],
                                  **_VET_ExtensionArguments )
                                           
    externalModules = cythonize([_VETLibExtension])                                       
else:
    _VETLibExtension = Extension( "pyVET._VET", 
                                  sources = ['pyVET/_VET.c'],
                                  **_VET_ExtensionArguments )
    externalModules = [_VETLibExtension]


build_requires=['numpy']



setup(
    name='pyVET',
    version='1.0',
    author = "Andres Perez Hortal",
    author_email = "andresperezcba@gmail.com",
    packages=find_packages(),
    ext_modules = externalModules,
    #url='http://pypi.python.org/pypi/pyVET/',
    license='LICENSE.txt',
    description='Variational Echo Tracking Algorithm',
    long_description=open('README.rst').read(),
    classifiers=[
    'Development Status :: 5 - Production/Stable',    'Intended Audience :: Science/Research',
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



