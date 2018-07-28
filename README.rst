
============================================
pyVET -- Variational Echo Tracking Algorithm
============================================

 * UNDER DEVELOPMENT!!! *

The pyVET package provides a python implementation of the 
Variational Echo Tracking presented in Laroche and Zawadzki (1995).

This algorithm is used by the McGill Algorithm for Prediction by 
Lagrangian Extrapolation (MAPLE) described in Germann and Zawadzki (2002).



Installing PyVET
================

Dependencies
------------

The pyVET package need the following dependencies

* future
* numpy
* cython
* scipy
* skimage
* matplolib (for examples)


Install from source
-------------------


The latest version can be installed manually by downloading the sources from
https://github.com/aperezhortal/pyVET

Then, for a **global installation** run::

    python setup.py install
    
For `user installation`_::

    python setup.py install --user

.. _user installation: \
    https://docs.python.org/2/install/#alternate-installation-the-user-scheme
    
If you want to put it somewhere different than your system files, you can do::
    
    python setup.py install --prefix=/path/to/local/dir

IMPORTANT: All the dependencies need to be already installed! 

