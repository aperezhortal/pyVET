======================================================
        pyVET -- Variational Echo Tracking Algorithm
        ======================================================
        
        pyVET provides a python implementation of the Variational Echo Tracking
        presented by Laroche and Zawazdki (1995).
    
        This algorithm is used by the McGill Algorithm for Prediction by Lagrangian Extrapolation (MAPLE)
        described in Germann and Zawadzki (2002).
        
        
        Notes
        -----
        Laroche, S., and I. Zawadzki, 1995: 
        Retrievals of horizontal winds from single-Doppler clear-air data by methods of 
        cross-correlation and variational analysis. J. Atmos. Oceanic Technol., 12, 721–738.
        doi: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2
        url: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2
     
        Germann, U. and I. Zawadzki, 2002: 
        Scale-Dependence of the Predictability of Precipitation from Continental Radar Images.
        Part I: Description of the Methodology. Mon. Wea. Rev., 130, 2859–2873,
        doi: 10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2.


*Technology builds on technology*

Documentation
=============

Check the documentation at http://www.meteo.mcgill.ca/~aperez/pyVET/

Dependencies
============

The SkewTplus package need the following dependencies

* numpy
* cython




Installing SkewT
================

PIP install
-----------

The easiest way is to install the package with pip::

    pip install pyVET


Install from source
-------------------

The latest version can be installed manually by downloading the sources from
https://github.com/aperezhortal/SkewTplus

Then, run::

    python setup.py install

If you want to put it somewhere different than your system files, you can do::
    
    python setup.py install --prefix=/path/to/local/dir

IMPORTANT: All the dependencies need to be already installed! 






