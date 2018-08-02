# -*- coding: utf-8 -*-

#
# Licensed under the BSD-3-Clause license
# Copyright (c) 2018, Andres A. Perez Hortal
#

"""
Cython module for morphing and cost functions implementations used in
in the Variation Echo Tracking Algorithm
"""


from cython.parallel import prange, parallel
from pyVET.error_handling import FatalError
import numpy as np

cimport cython
cimport numpy as np

ctypedef np.float64_t float64

from libc.math cimport floor, round

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np


cdef inline float64 float_abs(float64 a) nogil: return a if a > 0. else -a
""" Return the absolute value of a float """

@cython.cdivision(True)
cdef inline float64 _linear_interpolation(float64 x,
                                          float64 x1,
                                          float64 x2,
                                          float64 y1,
                                          float64 y2) nogil:
    """
    Linear interpolation at x.
    y(x) = y1 + (x-x1) * (y2-y1) / (x2-x1)
    """
    
    if (float_abs(x1 - x2) < 1e-6):
        return y1
        
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1) 

@cython.cdivision(True)
cdef inline float64 _bilinear_interpolation(float64 x,
                                            float64 y,
                                            float64 x1,
                                            float64 x2,
                                            float64 y1,
                                            float64 y2,
                                            float64 Q11,
                                            float64 Q12,
                                            float64 Q21,
                                            float64 Q22) nogil:
    '''https://en.wikipedia.org/wiki/Bilinear_interpolation'''
    
    cdef float64 f_x_y1, f_x_y2
    
    f_x_y1 = _linear_interpolation(x, x1, x2, Q11, Q21)
    f_x_y2 = _linear_interpolation(x, x1, x2, Q12, Q22)
    return _linear_interpolation(y, y1, y2, f_x_y1, f_x_y2)



    
# TODO: Add linear_interpolation_1D , nogil 
# TODO: Add linear_interpolation_2D , nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# TODO rename to WARP
def _morph(np.ndarray[float64, ndim=2] image,
           np.ndarray[float64, ndim=3] displacement):
    """ 
    
    Morph image by applying a displacement field (Warping).
    
    The new image is created by selecting for each position the values of the
    input image at the positions given by the x and y displacements.  
    The routine works in a backward sense. 
    The displacement vectors have to refer to their destination.
    
    For more information in Morphing functions see Section 3 in 
    `Beezley and Mandel (2008)`_.
        
    Beezley, J. D., & Mandel, J. (2008). 
    Morphing ensemble Kalman filters. Tellus A, 60(1), 131-140.
    
    .. _`Beezley and Mandel (2008)`: http://dx.doi.org/10.1111/\
    j.1600-0870.2007.00275.x

     
    The displacement field in x and y directions and the image must have the
    same dimensions.
    
    The morphing is executed in parallel over x axis.
    
    The value of displaced pixels that fall outside the limits takes the 
    value of the nearest edge. Those pixels are indicated by values greater
    than 1 in the output mask.
    
    Parameters
    ----------
    
    image : ndarray (ndim = 2)
        Image to morph
    
    displacement : ndarray (ndim = 3)
        Displacement field to be applied (Warping). 
        
        The dimensions are:
        displacement [ x (0) or y (1) , 
                      i index of pixel, j index of pixel ]
                      
    Returns
    -------
    
    morphed_image : ndarray (float64 ,ndim = 2)
        Morphed image
    
    mask : ndarray (int8 ,ndim = 2)
        Invalid values mask. Points outside the boundaries are masked.
        Values greater than 1, indicate masked values. 
    """
    # image(y,x) Rows y  , columns x    
    cdef np.intp_t nx = < np.intp_t > image.shape[0]      
    cdef np.intp_t ny = < np.intp_t > image.shape[1]
          
    # Use bilinear interpolation to obtain the value of the displaced pixel
    
    cdef np.ndarray[float64, ndim = 2] new_image = (
        np.zeros([nx, ny], dtype=np.float64))
    
    cdef np.ndarray[np.int8_t, ndim = 2] mask = (
        np.zeros([nx, ny], dtype=np.int8))
   
    cdef np.intp_t x, y
    
    cdef np.intp_t x_max_value_int = nx - 1
    cdef np.intp_t y_max_value_int = ny - 1

    cdef float64 xMaxValuefloat = < float64 > x_max_value_int
    cdef float64 yMaxValuefloat = < float64 > y_max_value_int
        
    cdef float64 x_float_value, y_float_value
    
    cdef float64 x_interpolation0
    cdef float64 x_interpolation1  
     
    cdef np.intp_t x_floor_int 
    cdef np.intp_t x_ceil_int
    cdef np.intp_t y_floor_int
    cdef np.intp_t y_ceil_int
    
    with nogil, parallel():
        for x in prange(nx , schedule='dynamic'):
            for y in range(ny):                
                
                x_float_value = (< float64 > x) - displacement[0, x, y] 
                y_float_value = (< float64 > y) - displacement[1, x, y]
                                
                if x_float_value < 0:
                    mask[x, y] = 1
                    x_float_value = 0
                    x_floor_int = 0
                    x_ceil_int = 0                    
                    
                elif x_float_value > xMaxValuefloat:
                    mask[x, y] = 1
                    x_float_value = xMaxValuefloat
                    x_floor_int = x_max_value_int
                    x_ceil_int = x_max_value_int                    
                    
                else:
                    x_floor_int = < np.intp_t > floor(x_float_value)
                    x_ceil_int = x_floor_int + 1
                    if x_ceil_int > x_max_value_int:
                        x_ceil_int = x_max_value_int
                                        
                if y_float_value < 0:
                    mask[x, y] = 1
                    y_float_value = 0
                    y_floor_int = 0
                    y_ceil_int = 0
                elif y_float_value > yMaxValuefloat:
                    mask[x, y] = 1
                    y_float_value = yMaxValuefloat
                    y_floor_int = y_max_value_int
                    y_ceil_int = y_max_value_int
                else:
                    y_floor_int = < np.intp_t > floor(y_float_value)
                    y_ceil_int = y_floor_int + 1
                    if y_ceil_int > y_max_value_int:
                        y_ceil_int = y_max_value_int
                
                    
 
                new_image[x, y] = _bilinear_interpolation(
                                        x_float_value,
                                        y_float_value,
                                        < float64 > x_floor_int,
                                        < float64 > x_ceil_int,
                                        < float64 > y_floor_int,
                                        < float64 > y_ceil_int,
                                        image[x_floor_int, y_floor_int],
                                        image[x_floor_int, y_ceil_int],
                                        image[x_ceil_int, y_floor_int],
                                        image[x_ceil_int, y_ceil_int])

    return new_image, mask


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _cost_function(np.ndarray[float64, ndim=3] sector_displacement,
                  np.ndarray[float64, ndim=2] template_image,
                  np.ndarray[float64, ndim=2] input_image,
                  np.ndarray[np.int8_t, ndim=2] mask,
                  float smooth_gain):
    """
    Variational Echo Tracking Cost function.
    
    This function computes the Variational Echo Tracking (VET) 
    Cost function presented  by `Laroche and Zawazdki (1995)`_ and used in the 
    McGill Algorithm for Prediction by Lagrangian Extrapolation (MAPLE) 
    described in
    `Germann and Zawadzki (2002)`_.
    
    
    .. _`Laroche and Zawazdki (1995)`: \
    http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2
    
    .. _`Germann and Zawadzki (2002)`: \
    http://dx.doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2
     
     
    The cost function is a the sum of the residuals of the squared image 
    differences along with a smoothness constrain.   
        
    This cost function implementation, supports displacement vector 
    sectorization.
    The displacement vector represent the displacement aaplied to the pixels in 
    each individual sector.
     
    This help to reduce the number of degrees of freedom of the cost function 
    when hierarchical approaches are used to obtain the minima of 
    the cost function (from low resolution to full image resolution).
    For example, in the MAPLE algorithm an Scaling Guess procedure is used to 
    find the displacement vectors.
    The echo motion field is retrieved in three runs with increasing resolution.
    The retrieval starts with (left) a uniform field, which is used as a first 
    guess to retrieve (middle) the field on a 5 × 5 grid, which in turn is the 
    first guess of (right) the final minimization with a 25 × 25 grid
    
    The shape of the sector is deduced from the image shape and the displacement
    vector shape. 
    
    IMPORTANT: The number of sectors in each dimension (x and y) must be a 
    factor full image size.
         
    The value of displaced pixels that fall outside the limits takes the 
    value of the nearest edge.
    
    The cost function is computed in parallel over the x axis.
    
    .. _ndarray: \
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
         
    Parameters
    ----------
    
    sector_displacement : ndarray_ (ndim=3)  
        Array of displacements to apply to each sector. The dimensions are:
        sector_displacement [ x (0) or y (1) displacement, 
                               i index of sector, j index of sector ]  
        
        
    template_image : ndarray_  (ndim=2)
        Input image array where the sector displacement is applied
     
    input_image : ndarray_
        Image array to be used as reference 
    
    smooth_gain : float
        Smoothness constrain gain
        
    Returns
    -------
    
    penalty : float
        Value of the cost function
    
    
    References
    ----------
    
    Laroche, S., and I. Zawadzki, 1995: 
    Retrievals of horizontal winds from single-Doppler clear-air data by methods
    of cross-correlation and variational analysis. 
    J. Atmos. Oceanic Technol., 12, 721–738.
    doi: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2
 
    Germann, U. and I. Zawadzki, 2002: 
    Scale-Dependence of the Predictability of Precipitation from Continental 
    Radar Images.
    Part I: Description of the Methodology. Mon. Wea. Rev., 130, 2859–2873,
    doi: 10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2. 
    
    """
    
    cdef np.intp_t x_sectors = < np.intp_t > sector_displacement.shape[1]
    cdef np.intp_t y_sectors = < np.intp_t > sector_displacement.shape[2]   
    
    cdef np.intp_t x_image_size = < np.intp_t > template_image.shape[0]
    cdef np.intp_t y_image_size = < np.intp_t > template_image.shape[1]
    
    if x_image_size % x_sectors != 0:
        raise FatalError("Error computing cost function",
                          "The number of sectors in x axis (axis=0)"
                          + " don't divide the image size")
    
    if y_image_size % y_sectors != 0:
        raise FatalError("Error computing cost function",
                          "The number of sectors in y axis (axis=1) don't"
                          + " divide the image size")
    
    cdef np.intp_t x_block_size = (
            < np.intp_t > (round(x_image_size / x_sectors)))
    
    cdef np.intp_t y_block_size = (
            < np.intp_t > (round(y_image_size / y_sectors)))
    
    
    cdef np.ndarray[float64, ndim = 3] real_displacement = (
            np.zeros([2, x_image_size, y_image_size], dtype=np.float64))
     
    cdef np.intp_t  x, y, i, j, l, m
    
    
    with nogil, parallel():
        for i in prange(x_sectors , schedule='dynamic'):            
            for j in range(y_sectors):                
                for l in range(x_block_size):
                    for m in range(y_block_size):
                                                
                        x = l + i * x_block_size
                        y = m + j * y_block_size
                                                                  
                        real_displacement[0, x, y] = (
                            sector_displacement[0, i, j])
                        real_displacement[1, x, y] = (
                            sector_displacement[1, i, j])
                        
            
    cdef np.ndarray[float64, ndim = 2] morphed_image 
    cdef np.ndarray[np.int8_t, ndim = 2] morph_mask
    
    morphed_image, morph_mask = _morph(template_image, real_displacement)
            
    cdef float64 residuals = 0 
    for x in range(x_image_size):
        for y in range(y_image_size):
            if (mask[x, y] == 0) and (morph_mask[x, y] == 0):
                residuals += (morphed_image[x, y] - input_image[x, y]) ** 2

            
    cdef float64 smoothness_penalty = 0
    
    cdef float64 df_dx2 = 0
    cdef float64 df_dxdy = 0
    cdef float64 df_dy2 = 0

    cdef float64 inloop_smoothness_penalty  
    
     
    if smooth_gain > 0.:
        
        for i in range(2):
            
            inloop_smoothness_penalty = 0
            
            with nogil, parallel():
                
                for x in prange(1, x_sectors - 1, schedule='dynamic'):
                
                    for y in range(1, y_sectors - 1):
                        
                        df_dx2 = (sector_displacement[i, x + 1, y] 
                                  - 2 * sector_displacement[i, x, y] 
                                  + sector_displacement[i, x - 1, y])
                        
                        df_dx2 = df_dx2 / (x_block_size * x_block_size)
                        
                        df_dy2 = (sector_displacement[i, x, y + 1] 
                                  - 2 * sector_displacement[i, x, y] 
                                  + sector_displacement[i, x, y - 1])
                         
                        df_dy2 = df_dy2 / (y_block_size * y_block_size)
                        
                        df_dxdy = (sector_displacement[i, x + 1, y + 1] 
                                   - sector_displacement[i, x + 1, y - 1] 
                                   - sector_displacement[i, x - 1, y + 1] 
                                   + sector_displacement[i, x - 1, y - 1])
                        df_dxdy = df_dxdy / (4 * x_block_size * y_block_size)
                                                       
                        inloop_smoothness_penalty = (df_dx2 * df_dx2 
                                                     + 2 * df_dxdy * df_dxdy
                                                     + df_dy2 * df_dy2)
            
                        smoothness_penalty += inloop_smoothness_penalty 
                                                
        smoothness_penalty *= smooth_gain * x_block_size * y_block_size            
                       
    return  residuals, smoothness_penalty

