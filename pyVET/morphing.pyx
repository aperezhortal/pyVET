from __future__ import division

from cython.parallel import prange, parallel
import numpy as np


cimport cython

from libc.math cimport floor, round

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np


@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
@cython.cdivision( True )
def _morph( np.ndarray[np.float64_t, ndim = 2] image, np.ndarray[np.float64_t, ndim = 3] displacement ):
    """ 
    
    Morph image applying a displacement field.
    
    Create a new image by selecting for each position the values of the
    input image at the positions given by the xdis(placement) and ydis(placement).
        
    The routine works in a backward sense. 
    The displacement vectors have to refer to their destination.
    
     
    The displacement field in x and y directions and the image must have the same dimensions
    
    Parameters
    ----------
    
    image : ndarray (ndim = 2)
        Image to morph
    
    displacement : ndarray (ndim = 3)
        Displacement field. 
        
        The dimensions are:
        displacement [ x (0) or y (1) dimension, 
                      i index of pixel, j index of pixel ]
                      
    Returns
    -------
    
    morphedImage : ndarray (ndim = 2)
        Morphed image
    """
    # image(y,x) Rows y  , columns x    
    cdef np.intp_t nx = < np.intp_t > image.shape[0]      
    cdef np.intp_t ny = < np.intp_t > image.shape[1]
          
    # Use bilinear interpolation to obtain the value of the displaced pixel
    
    cdef np.ndarray[np.float64_t, ndim = 2] newImage = np.zeros( [nx, ny], dtype = np.float64 )
    
   
    cdef np.intp_t x, y
    
    cdef np.intp_t xMaxValueInt = nx - 1
    cdef np.intp_t yMaxValueInt = ny - 1

    cdef np.float64_t xMaxValuefloat = < np.float64_t > xMaxValueInt
    cdef np.float64_t yMaxValuefloat = < np.float64_t > yMaxValueInt
        
    cdef np.float64_t xfloatValue, yfloatValue
    cdef np.float64_t xInterpolation0, xInterpolation1  
     
    cdef np.intp_t xFloorInt, xCeilInt, yFloorInt, yCeilInt 

    with nogil, parallel():
        for x in prange( nx , schedule = 'dynamic' ):
            for y in range( ny ):                
                
                xfloatValue = < np.float64_t > x - displacement[0, x, y] 
                yfloatValue = < np.float64_t > y - displacement[1, x, y]
                                
                if xfloatValue < 0:
                    xfloatValue = 0
                    xFloorInt = 0
                    xCeilInt = 0                    
                    
                elif xfloatValue > xMaxValuefloat:
                    xfloatValue = xMaxValuefloat
                    xFloorInt = xMaxValueInt
                    xCeilInt = xMaxValueInt                    
                    
                else:
                    xFloorInt = < np.intp_t > floor( xfloatValue )
                    xCeilInt = xFloorInt + 1
                    if xCeilInt > xMaxValueInt:
                        xCeilInt = xMaxValueInt
                    
                    
                if yfloatValue < 0:
                    yfloatValue = 0
                    yFloorInt = 0
                    yCeilInt = 0
                elif yfloatValue > yMaxValuefloat:
                    yfloatValue = yMaxValuefloat
                    yFloorInt = yMaxValueInt
                    yCeilInt = yMaxValueInt
                else:
                    yFloorInt = < np.intp_t > floor( yfloatValue )
                    yCeilInt = yFloorInt + 1
                    if yCeilInt > yMaxValueInt:
                        yCeilInt = yMaxValueInt
                
                    
                if xFloorInt == xCeilInt:
                    
                    xInterpolation0 = image[xFloorInt, yFloorInt]
                    xInterpolation1 = image[xFloorInt, yCeilInt]                    
    
                else:
                    
                    xInterpolation0 = ( ( < np.float64_t > xCeilInt - xfloatValue ) * image[xFloorInt, yFloorInt] + 
                                        ( xfloatValue -< np.float64_t > xFloorInt ) * image[xCeilInt, yFloorInt] )
                    xInterpolation1 = ( ( < np.float64_t > xCeilInt - xfloatValue ) * image[xFloorInt, yCeilInt] + 
                                        ( xfloatValue -< np.float64_t > xFloorInt ) * image[xCeilInt, yCeilInt] )
                    
                    
                
                if xInterpolation0 == xInterpolation1:
                    newImage[x, y] = xInterpolation0
                else:
                    newImage[x, y] = ( xInterpolation0 * ( < np.float64_t > yCeilInt - yfloatValue ) + 
                                      xInterpolation1 * ( yfloatValue -< np.float64_t > yFloorInt ) )
                
    
    return newImage


@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
@cython.cdivision( True )
def _costFunction( np.ndarray[np.float64_t, ndim = 3] sectorDisplacement2D,
                   np.ndarray[np.float64_t, ndim = 2] inputImage,
                   np.ndarray[np.float64_t, ndim = 2] referenceImage,
                   float smoothGain ):
    """
    Displacement cost function
    
    The cost function is a the sum of the residuals of the squared image differences 
    and a smoothness constrain.
    
    The displacement vector represent the displacement in each sector.
    The shape of the sector is deduced from the image shape and the displacement 
    vector shape.
    
    Inside each sector, the same displacement is applied to each grid point.
        
    The displaced pixels that fall outside the limits takes the 
    value of the nearest edge.
         
    Parameters
    ----------
    
    sectorDisplacement2D : ndarray (ndim=3)  
        Array of displacements to apply to each sector. The dimensions are:
        sectorDisplacement2D [ x (0) or y (1) displacement, 
                               i index of sector, j index of sector ]  
        
        
    inputImage : ndarray  (ndim=2)
        Input image array where the sector displacement is applied
     
    referenceImage : ndarray
        Image array to be used as reference 
    
    smoothGain : float
        Smothness constrain gain
        
    Returns
    -------
    
    penalty : float
    """
    
    cdef np.intp_t sectorsInX = < np.intp_t > sectorDisplacement2D.shape[1]
    cdef np.intp_t sectorsInY = < np.intp_t > sectorDisplacement2D.shape[2]   
    
    cdef np.intp_t imageSizeInX = < np.intp_t > inputImage.shape[0]
    cdef np.intp_t imageSizeInY = < np.intp_t > inputImage.shape[1]
    
    if imageSizeInX % sectorsInX != 0:
        raise Exception( "The number of sectors in axis=0 don't divide the image size" )
    
    if imageSizeInY % sectorsInY != 0:
        raise Exception( "The number of sectors in axis=0 don't divide the image size" )

    cdef np.intp_t blockSizeInX = < np.intp_t > ( round( imageSizeInX / sectorsInX ) )
    cdef np.intp_t blockSizeInY = < np.intp_t > ( round( imageSizeInY / sectorsInY ) )
    
    
    cdef np.ndarray[np.float64_t, ndim = 3] realDisplacement = np.zeros( [2, imageSizeInX, imageSizeInY],
                                                                         dtype = np.float64 )
     
    cdef np.intp_t  x, y, i, j, l, m
    
    
    with nogil, parallel():
        for i in prange( sectorsInX , schedule = 'dynamic' ):            
            for j in range( sectorsInY ):                
                for l in range( blockSizeInX ):
                    for m in range( blockSizeInY ):
                                                
                        x = l + i * blockSizeInX
                        y = m + j * blockSizeInY
                                                                  
                        realDisplacement[0, x, y] = sectorDisplacement2D[0, i, j]
                        realDisplacement[1, x, y] = sectorDisplacement2D[1, i, j]
                        
                    
                
            
    cdef np.ndarray[np.float64_t, ndim = 2] morphedImage = _morph( inputImage, realDisplacement )
            
    cdef np.float64_t residuals
    residuals = 0 
    for x in range( imageSizeInX ):
        for y in range( imageSizeInY ):
            residuals += ( morphedImage[x, y] - referenceImage[x, y] ) ** 2

            
    cdef np.float64_t smoothnessPenalty = 0
    
    cdef  np.float64_t sectorSmoothGain[2]
    cdef  np.float64_t tempSmoothPenalty  
    
     
    if smoothGain > 0.:
        
        sectorSmoothGain[0] = smoothGain / ( < np.float64_t > ( blockSizeInX * blockSizeInX ) )  
        sectorSmoothGain[1] = smoothGain / ( < np.float64_t > ( blockSizeInY * blockSizeInY ) )
        
        for i in range( 2 ):
            
            tempSmoothPenalty = 0
            
            with nogil, parallel():
                
                for x in prange( 1, sectorsInX - 1, schedule = 'dynamic' ):
                
                    for y in range( 1, sectorsInY - 1 ):
                                            
                        tempSmoothPenalty += ( ( sectorDisplacement2D[i, x + 1, y] - 2 * sectorDisplacement2D[i, x, y] + sectorDisplacement2D[i, x - 1, y] ) ** 2 + 
                                               ( sectorDisplacement2D[i, x, y + 1] - 2 * sectorDisplacement2D[i, x, y] + sectorDisplacement2D[i, x, y - 1] ) ** 2 + 
                                               0.125 * ( sectorDisplacement2D[i, x + 1, y + 1] - sectorDisplacement2D[i, x + 1, y - 1] - sectorDisplacement2D[i, x - 1, y + 1] + 
                                                sectorDisplacement2D[i, x - 1, y - 1] ) ** 2 )  
            
            smoothnessPenalty += tempSmoothPenalty * sectorSmoothGain[i]            
                       
    return  residuals + smoothnessPenalty

