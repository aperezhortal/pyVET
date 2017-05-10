'''
Variational Echo Tracking (VET) Module


This module implements the VET algorithm presented
by `Laroche and Zawazdki (1995)`_ and used in the 
McGill Algorithm for Prediction by Lagrangian Extrapolation (MAPLE) described in
`Germann and Zawadzki (2002)`_.


.. _`Laroche and Zawazdki (1995)`: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

.. _`Germann and Zawadzki (2002)`: http://dx.doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2
     

The morphing and the cost functions are implemented in Cython and paralellized for performance
purposes.
   
References
----------    

Laroche, S., and I. Zawadzki, 1995: 
Retrievals of horizontal winds from single-Doppler clear-air data by methods of 
cross-correlation and variational analysis. J. Atmos. Oceanic Technol., 12, 721–738.
doi: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

Germann, U. and I. Zawadzki, 2002: 
Scale-Dependence of the Predictability of Precipitation from Continental Radar Images.
Part I: Description of the Methodology. Mon. Wea. Rev., 130, 2859–2873,
doi: 10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2.      



'''
# For python 3 portability
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy
from scipy.ndimage.interpolation import zoom
from scipy.optimize._minimize import minimize
from skimage.util.shape import view_as_blocks

from pyVET._VET import _costFunction
from pyVET.errorHandling import fatalError


__author__ = "Andres Perez Hortal"
__copyright__ = "Copyright (c) 2017, Andres A. Perez Hortal, McGill University"
__license__ = "BSD-3-Clause License, see LICENCE.txt for more details"
__email__ = "andresperezcba@gmail.com"







def roundInt( scalar ):
    """
    Round number to nearest integer. Returns and integer value.
    """
    return int( round( scalar ) )



def costFunction( sectorDisplacement1D, inputImage, referenceImage, blocksShape, smoothGain ):
    """
    Variational Echo Tracking Cost Function
    
    .. _`scipy.optimize.minimize` : https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html
    
    This function is designed to be used with the `scipy.optimize.minimize`_ 
    
    The function first argument is the variable to be used in the minimization procedure.
    
    The sector displacement must be a flat array compatible with the dimensions of the
    input image and sectors shape (see parameters section below for more details).
        
    
    .. _ndarray: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
    
    Parameters
    ----------
    
    
    sectorDisplacement1D : ndarray_ 
        Array of displacements to apply to each sector. The dimensions are:
        sectorDisplacement2D [ x (0) or y (1) displacement, 
                               i index of sector, j index of sector ].
        The shape of the sector displacements must be compatible with the 
        input image and the block shape.
        The shape should be (2, mx, my) where mx and my are the numbers of 
        sectors in the x and the y dimension.
        
                                 
        
        
    inputImage : ndarray_  (ndim=2)
        Input image array (nx by ny pixels) where the sector displacement is applied.
        
     
    referenceImage : ndarray_ (ndim=2)
        Image array to be used as reference (nx by ny pixels).
    
    blocksShape : ndarray_ (ndim=2)
        Number of sectors in each dimension (x and y).
        blocksShape.shape = (mx,my)
         
            
    smoothGain : float
        Smoothness constrain gain
        
    
    Returns
    -------
    
    penalty : float
        Value of the cost function
    """    
    
    sectorDisplacement2D = sectorDisplacement1D.reshape( *( ( 2, ) + blocksShape ) )
    
    return _costFunction( sectorDisplacement2D, inputImage, referenceImage, smoothGain )



#TODO input parameter check
def VET( inputImage, referenceImage, 
         factors = [ 64, 16, 4, 2, 1 ], 
         smoothGain = 100,
         firstGuess = None,
         verbose=False ):
    '''
    Variational Echo Tracking Algorithm presented in `Laroche and Zawazdki (1995)`_ 
    and used in the McGill Algorithm for Prediction by Lagrangian Extrapolation (MAPLE) described in
    `Germann and Zawadzk (2002)`_.

    .. _`Laroche and Zawazdki (1995)`: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

    .. _`Germann and Zawadzki (2002)`: http://dx.doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2
     
    
    This algorithm computes the displacement field (Warping) between two images
    that minimizes sum of the residuals of the squared image differences along with
    a smoothness constrain.   
    
    In order to find the minimum an scaling guess procedure is applied, from larger scales
    to a finer scale. This reduces the changes that the minimization procedure converges to 
    a local minima. The scaling guess is defined by the scaling factors (see **factors** keyword).
    
    The smoothness of the returned displacement field is controlled by the smoothness constrain
    gain (**smoothGain** keyword).
    
    If a first guess is not given, zero displacements are used as first guess. 
    
    
    Parameters
    ----------
    
    inputImage : ndarray_
        Input image of nx by ny pixels (2D array) 

    referenceImage : ndarray_
        Reference image used to obtain the displacement (2D array).
        Same shape as the input image.
            
    factors : list or array 
        If dimension is 1, the same factors will be used both image dimensions (x and y)
        If is 2D, the each row determines the factors of the corresponding dimensions.
        The factors denotes the number of sectors in each dimension used in each scaling procedure.
        
    smoothGain : float
        Smooth gain factor
        
    firstGuess : ndarray_
        If firstGuess is not present zeros are used as first guess.
        The shape should be compatible with the input image.
        That is (2,nx,ny).
        
    verbose : bool
        Verbosity enabled if True
    
    
    Returns
    -------
    
    displacementField : ndarray_
        Displacement Field (2D array) 
    
    References
    ----------    
    
    Laroche, S., and I. Zawadzki, 1995: 
    Retrievals of horizontal winds from single-Doppler clear-air data by methods of 
    cross-correlation and variational analysis. J. Atmos. Oceanic Technol., 12, 721–738.
    doi: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2
    
    Germann, U. and I. Zawadzki, 2002: 
    Scale-Dependence of the Predictability of Precipitation from Continental Radar Images.
    Part I: Description of the Methodology. Mon. Wea. Rev., 130, 2859–2873,
    doi: 10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2. 
    
    '''
    
    if verbose:
        def debugPrint(*args,**kwargs):
            print(*args,**kwargs)
    else:
        def debugPrint(*args,**kwargs):
            pass
        
         
    inputImage = numpy.array( inputImage, dtype = numpy.float64 )
    referenceImage = numpy.array( referenceImage, dtype = numpy.float64 )
    
    
    # Check that the factors divide the domain
    factors = numpy.array( factors, dtype = numpy.int )
    
    if factors.ndim == 1 and inputImage.shape[0] == inputImage.shape[1]:
        newFactors = numpy.zeros( ( 2, ) + factors.shape ) + factors.reshape( ( 1, factors.shape[0] ) )
        factors = newFactors
        
    if factors.ndim != 2:
        if factors.shape[0] != 2:
            raise fatalError( "Error computing VET",
                              "Incorrect factors dimensions.",
                              "factor should be a 1D or 2D array" )
    
    # Check that the factors divide the domain    
    for i in range( factors.shape[1] ):
        
        if ( inputImage.shape[0] % factors[0, i] ) > 0:
            raise Exception( "The factor %d does not divide x dimension" % factors[0, i] )
        
        if ( inputImage.shape[1] % factors[1, i] ) > 0:
            raise Exception( "The factor %d does not divide y dimension" % factors[1, i] )
    
    
    # Sort factors in descending order 
    factors.sort(axis=1)
    debugPrint(factors)
    # Prepare first guest
    if firstGuess is None:
        firstGuess = numpy.zeros( ( 2, ) + inputImage.shape, order = 'C' )
    else:
        if firstGuess.shape != ( 2, ) + inputImage.shape:
            raise fatalError( "The shape of the initial guess do not match the Input image dimensions" )


    for i in range( factors.shape[1] ):        
        # Minimize for each sector size
        
        sectorShape = ( roundInt( inputImage.shape[0] / factors[0, i] ),
                        roundInt( inputImage.shape[1] / factors[1, i] ) )
        
        debugPrint("\nSector Shape:",sectorShape)
        
        sectorFirstGuess = view_as_blocks( firstGuess,
                                           block_shape = ( 1, ) + sectorShape )
        
        blocksShape = sectorFirstGuess.shape[1:3]
        
#         import cProfile, pstats, StringIO
#         pr = cProfile.Profile()
#         pr.enable()
        
        initialGuess = numpy.mean( numpy.mean( sectorFirstGuess,
                                               axis = 4, keepdims = True ),
                                               axis = 5, keepdims = True )
        debugPrint("Minimizing")
        result = minimize( costFunction,
                           initialGuess[:, :, :, 0, 0, 0].flatten(),
                           args = ( inputImage, referenceImage, blocksShape,smoothGain ),
                           method = 'CG', options = {'eps':0.1, 'gtol': 0.1, 'maxiter' : 20, 'disp': True}
                           ) 
#         pr.disable()
#         s = StringIO.StringIO()
#         sortby = 'tottime'
#         ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#         ps.print_stats()
#         print s.getvalue()

        sectorDisplacements = result.x
        if factors[0, i] > 1 and factors[1, i] > 1:
            
            sectorDisplacement2D = sectorDisplacements.reshape( *( sectorFirstGuess.shape[0:3] ) )
            
            zoomFactor = ( 1, sectorFirstGuess.shape[4], sectorFirstGuess.shape[5] )
            
            firstGuess = zoom( sectorDisplacement2D, zoomFactor, order = 1, mode = 'nearest' )
            
            
        else:
            firstGuessAsBlocks = view_as_blocks( firstGuess,
                                                 block_shape = ( 1, ) + sectorShape )    

            firstGuessAsBlocks[:] = sectorDisplacements.reshape( ( 2, ) + sectorFirstGuess.shape[1:3] + ( 1, 1, 1 ) )
        
    return firstGuess 




