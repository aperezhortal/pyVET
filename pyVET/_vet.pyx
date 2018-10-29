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
import numpy as np

cimport cython
cimport numpy as np

ctypedef np.float64_t float64
ctypedef np.int8_t int8

from libc.math cimport floor, round

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

    if float_abs(x1 - x2) < 1e-6:
        return y1

    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

@cython.cdivision(True)
cdef inline float64 _bilinear_interpolation(float64 x,
                                            float64 y,
                                            float64 x1,
                                            float64 x2,
                                            float64 y1,
                                            float64 y2,
                                            float64 q11,
                                            float64 q12,
                                            float64 q21,
                                            float64 q22) nogil:
    """https://en.wikipedia.org/wiki/Bilinear_interpolation"""

    cdef float64 f_x_y1, f_x_y2

    f_x_y1 = _linear_interpolation(x, x1, x2, q11, q21)
    f_x_y2 = _linear_interpolation(x, x1, x2, q12, q22)
    return _linear_interpolation(y, y1, y2, f_x_y1, f_x_y2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _warp(np.ndarray[float64, ndim=2] image,
          np.ndarray[float64, ndim=3] displacement,
          bint gradient=False):
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

    gradient : bool, optional
        If True, the gradient of the morphing function is returned.


    Returns
    -------
    
    image : ndarray (float64 ,ndim = 2)
        Morphed image.
    
    mask : ndarray (int8 ,ndim = 2)
        Invalid values mask. Points outside the boundaries are masked.
        Values greater than 1, indicate masked values.

    gradient_values : ndarray (float64 ,ndim = 3), optional
        If gradient keyword is True, the gradient of the function is also
        returned.
    """

    cdef np.intp_t nx = <np.intp_t> image.shape[0]
    cdef np.intp_t ny = <np.intp_t> image.shape[1]

    cdef np.ndarray[float64, ndim = 2] new_image = (
        np.zeros([nx, ny], dtype=np.float64))

    cdef np.ndarray[int8, ndim = 2] mask = (
        np.zeros([nx, ny], dtype=np.int8))

    cdef np.ndarray[float64, ndim = 3] gradient_values = (
        np.zeros([2, nx, ny], dtype=np.float64))

    cdef np.intp_t x, y

    cdef np.intp_t x_max_int = nx - 1
    cdef np.intp_t y_max_int = ny - 1

    cdef float64 x_max_float = <float64> x_max_int
    cdef float64 y_max_float = <float64> y_max_int

    cdef float64 x_float, y_float, dx, dy

    cdef np.intp_t x_floor
    cdef np.intp_t x_ceil
    cdef np.intp_t y_floor
    cdef np.intp_t y_ceil

    cdef float64 f00, f10, f01, f11

    for x in prange(nx, schedule='dynamic', nogil=True):
        for y in range(ny):

            x_float = (<float64> x) - displacement[0, x, y]
            y_float = (<float64> y) - displacement[1, x, y]

            if x_float < 0:
                mask[x, y] = 1
                x_float = 0
                x_floor = 0
                x_ceil = 0

            elif x_float > x_max_float:
                mask[x, y] = 1
                x_float = x_max_float
                x_floor = x_max_int
                x_ceil = x_max_int

            else:
                x_floor = <np.intp_t> floor(x_float)
                x_ceil = x_floor + 1
                if x_ceil > x_max_int:
                    x_ceil = x_max_int

            if y_float < 0:
                mask[x, y] = 1
                y_float = 0
                y_floor = 0
                y_ceil = 0
            elif y_float > y_max_float:
                mask[x, y] = 1
                y_float = y_max_float
                y_floor = y_max_int
                y_ceil = y_max_int
            else:
                y_floor = <np.intp_t> floor(y_float)
                y_ceil = y_floor + 1
                if y_ceil > y_max_int:
                    y_ceil = y_max_int

            dx = x_float - <float64> x_floor
            dy = y_float - <float64> y_floor

            # This assumes that the spacing between grid points=1.

            # Bilinear interpolation coeficients
            f00 = image[x_floor, y_floor]
            f10 = image[x_ceil, y_floor] - image[x_floor, y_floor]
            f01 = image[x_floor, y_ceil] - image[x_floor, y_floor]
            f11 = (image[x_floor, y_floor] - image[x_ceil, y_floor]
                   - image[x_floor, y_ceil] + image[x_ceil, y_ceil])

            # Bilinear interpolation
            new_image[x, y] = f00 + dx * f10 + dy * f01 + dx * dy * f11

            if gradient:
                gradient_values[0, x, y] = f10 + dy * f11
                gradient_values[1, x, y] = f01 + dx * f11

    if gradient:
        return new_image, mask, gradient_values
    else:
        return new_image, mask

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _cost_function(np.ndarray[float64, ndim=3] sector_displacement,
                   np.ndarray[float64, ndim=2] template_image,
                   np.ndarray[float64, ndim=2] input_image,
                   np.ndarray[int8, ndim=2] mask,
                   float smooth_gain,
                   bint gradient = False):
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
    The displacement vector represent the displacement applied to the pixels in
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
        Input image array where the sector displacement is applied.
     
    input_image : ndarray_
        Image array to be used as reference 
    
    smooth_gain : float
        Smoothness constrain gain

    mask : ndarray_ (ndim=2)
        Data mask. If is True, the data is marked as not valid and is not
        used in the computations.

    gradient : bool, optional
        If True, the gradient of the morphing function is returned.

    Returns
    -------
    
    penalty or  gradient values.

    penalty : float
        Value of the cost function

    gradient_values : ndarray (float64 ,ndim = 3), optional
        If gradient keyword is True, the gradient of the function is also
        returned.
    
    
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

    cdef np.intp_t x_sectors = <np.intp_t> sector_displacement.shape[1]
    cdef np.intp_t y_sectors = <np.intp_t> sector_displacement.shape[2]

    cdef np.intp_t x_image_size = <np.intp_t> template_image.shape[0]
    cdef np.intp_t y_image_size = <np.intp_t> template_image.shape[1]

    if x_image_size % x_sectors != 0:
        raise ValueError("Error computing cost function.\n",
                         "The number of sectors in x axis (axis=0)"
                         + " don't divide the image size")

    if y_image_size % y_sectors != 0:
        raise ValueError("Error computing cost function.\n",
                         "The number of sectors in y axis (axis=1) don't"
                         + " divide the image size")

    cdef np.intp_t x_block_size = (
        <np.intp_t> (round(x_image_size / x_sectors)))

    cdef np.intp_t y_block_size = (
        <np.intp_t> (round(y_image_size / y_sectors)))

    cdef np.ndarray[float64, ndim = 3] real_displacement = (
        np.zeros([2, x_image_size, y_image_size], dtype=np.float64))

    cdef np.intp_t  i, j  # Original image indexes
    cdef np.intp_t  ii, jj  # Original image indexes
    cdef np.intp_t  l, m  # Sectors indexes
    cdef np.int_t axis  # x or y axis

    #Assume regular grid with constant grid spacing.
    _x = np.arange(x_image_size, dtype='float64')
    _y = np.arange(y_image_size, dtype='float64')

    cdef np.ndarray[float64, ndim = 2] x
    cdef np.ndarray[float64, ndim = 2] y

    x, y = np.meshgrid(_x, _y, indexing='ij')

    _xd = _x.reshape((x_sectors, x_block_size)).mean(axis=1)
    _yd = _y.reshape((y_sectors, y_block_size)).mean(axis=1)

    cdef np.ndarray[float64, ndim = 2] xd

    cdef np.ndarray[float64, ndim = 2] yd

    xd, yd = np.meshgrid(_xd, _yd, indexing='ij')

    del _xd, _yd, _x, _y

    # for i in prange(x_image_size, schedule='dynamic', nogil=False):
    for i in range(x_image_size):
        for j in range(y_image_size):

            l = i / x_sectors
            m = j / y_sectors

            if l >= (x_sectors - 1):
                l = x_sectors - 2

            if m >= (y_sectors - 1):
                m = y_sectors - 2

            for axis in range(2):
                real_displacement[axis, i, j] = _bilinear_interpolation(
                    x[i, j], y[i, j],
                    xd[l, m], xd[l + 1, m],
                    yd[l, m], yd[l, m + 1],
                    sector_displacement[axis, l, m],
                    sector_displacement[axis, l, m + 1],
                    sector_displacement[axis, l + 1, m],
                    sector_displacement[axis, l + 1, m + 1])

    cdef np.ndarray[float64, ndim = 2] morphed_image
    cdef np.ndarray[int8, ndim = 2] morph_mask
    cdef np.ndarray[float64, ndim = 3] _gradient_data
    cdef np.ndarray[float64, ndim = 3] gradient_residuals
    cdef np.ndarray[float64, ndim = 3] gradient_smooth

    cdef np.ndarray[float64, ndim = 2] buffer = \
        np.zeros([x_image_size, y_image_size], dtype=np.float64)

    cdef float64 residuals = 0

    # Compute residual part of the cost function
    if gradient:

        gradient_smooth = np.zeros([2, x_sectors, y_sectors],
                                   dtype=np.float64)

        gradient_residuals = np.zeros([2, x_sectors, y_sectors],
                                      dtype=np.float64)

        morphed_image, morph_mask, _gradient_data = _warp(template_image,
                                                          real_displacement,
                                                          gradient=True)
        mask[morph_mask == 1] = 1
        buffer = (2 * (input_image - morphed_image))

        # Sum over each sector.
        for l in prange(x_sectors, schedule='dynamic', nogil=True):
            for m in range(y_sectors):
                for ii in range(x_block_size):
                    for jj in range(y_block_size):

                        i = ii + m * x_block_size
                        j = jj + m * y_block_size

                        if mask[i, i] == 0:
                            gradient_residuals[0, l, m] += \
                                _gradient_data[0, i, j] * buffer[i, j]

                            gradient_residuals[1, l, m] += \
                                _gradient_data[1, i, i] * buffer[i, j]

    else:

        morphed_image, morph_mask = _warp(template_image,
                                          real_displacement,
                                          gradient=False)
        mask[morph_mask == 1] = 1
        residuals = np.sum((morphed_image - input_image)[mask == 0] ** 2)

    # Compute smoothness constraint part of the cost function
    cdef float64 smoothness_penalty = 0

    cdef float64 df_dx2 = 0
    cdef float64 df_dxdy = 0
    cdef float64 df_dy2 = 0

    cdef float64 inloop_smoothness_penalty

    if smooth_gain > 0.:

        smooth_gain  *= x_block_size * y_block_size

        for axis in range(2):

            inloop_smoothness_penalty = 0

            for i in prange(1, x_sectors - 1, schedule='dynamic', nogil=True):

                for j in range(1, y_sectors - 1):
                    df_dx2 = (sector_displacement[axis, i + 1, j]
                              - 2 * sector_displacement[axis, i, j]
                              + sector_displacement[axis, i - 1, j])

                    df_dx2 = df_dx2 / (x_block_size * x_block_size)

                    df_dy2 = (sector_displacement[axis, i, j + 1]
                              - 2 * sector_displacement[axis, i, j]
                              + sector_displacement[axis, i, j - 1])

                    df_dy2 = df_dy2 / (y_block_size * y_block_size)

                    df_dxdy = (sector_displacement[axis, i + 1, j + 1]
                               - sector_displacement[axis, i + 1, j - 1]
                               - sector_displacement[axis, i - 1, j + 1]
                               + sector_displacement[axis, i - 1, j - 1])

                    df_dxdy = df_dxdy / (4 * x_block_size * y_block_size)
                    # df_dxdy = df_dxdy * 0.25

                    if gradient:
                        gradient_smooth[axis, i, j] -= 2 * df_dx2
                        gradient_smooth[axis, i + 1, j] += df_dx2
                        gradient_smooth[axis, i - 1, j] += df_dx2

                        gradient_smooth[axis, i, j] -= 2 * df_dy2
                        gradient_smooth[axis, i, j - 1] += df_dy2
                        gradient_smooth[axis, i, j + 1] += df_dy2

                        gradient_smooth[axis, i - 1, j - 1] += df_dxdy
                        gradient_smooth[axis, i - 1, j + 1] -= df_dxdy
                        gradient_smooth[axis, i + 1, j - 1] -= df_dxdy
                        gradient_smooth[axis, i + 1, j + 1] += df_dxdy

                    inloop_smoothness_penalty = (df_dx2 **2
                                                 + 2 * df_dxdy**2
                                                 + df_dy2**2)

                    smoothness_penalty += inloop_smoothness_penalty

        smoothness_penalty *= smooth_gain

    if gradient:
        gradient_smooth *= 2 * smooth_gain

        return gradient_residuals + gradient_smooth
    else:
        return residuals, smoothness_penalty
