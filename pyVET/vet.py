# -*- coding: utf-8 -*-
#
# Licensed under the BSD-3-Clause license
# Copyright (c) 2018, Andres A. Perez Hortal
'''
Variational Echo Tracking (VET) Module


This module implements the VET algorithm presented
by `Laroche and Zawadzki (1995)`_ and used in the
McGill Algorithm for Prediction by Lagrangian Extrapolation (MAPLE) described
in `Germann and Zawadzki (2002)`_.


.. _`Laroche and Zawadzki (1995)`:\
    http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

.. _`Germann and Zawadzki (2002)`:\
    http://dx.doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2


The morphing and the cost functions are implemented in Cython and parallelized
for performance.
'''
# For python 3 portability
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from numpy import float64
import numpy
from numpy.ma.core import MaskedArray
from scipy.ndimage.interpolation import zoom
from scipy.optimize._minimize import minimize
from skimage.util.shape import view_as_blocks

from pyVET._vet import _cost_function  # @UnresolvedImport
from pyVET._vet import _morph  # @UnresolvedImport


def round_int(scalar):
    """
    Round number to nearest integer. Returns and integer value.
    """
    return int(round(scalar))


def morph(image, displacement):
    """
    Morph image by applying a displacement field (Warping) in the images
    coordinates.
    
    Take caution that (rows, columns)=(i,j)==(y,x).

    See :py:func:`vet._vet._morph`.
    """
    _image = numpy.asarray(image, dtype=float64, order='C')
    _displacement = numpy.asarray(displacement, dtype=float64, order='C')
    return _morph(_image, _displacement)


def downsize(input_array, x_factor, y_factor=None):
    '''
    Reduce resolution of an array by neighbourhood averaging (2D averaging)
    of x_factor by y_factor elements.

    Parameters
    ----------

    input_array: ndarray
        Array to downsize by neighbourhood averaging

    x_factor : int
        factor of downsizing in the x dimension

    y_factor : int
        factor of downsizing in the y dimension

    Returns
    -------
    '''

    x_factor = int(x_factor)

    if y_factor is None:
        y_factor = x_factor
    else:
        y_factor = int(y_factor)

    input_block_view = view_as_blocks(input_array, (x_factor, y_factor))

    data = input_block_view.mean(-1).mean(-1)

    return numpy.ma.masked_invalid(data)


def cost_function(sector_displacement_1d,
                  template_image,
                  input_image,
                  blocks_shape,
                  mask,
                  smooth_gain,
                  debug=False):
    """
    Variational Echo Tracking Cost Function

    .. _`scipy.optimize.minimize` :\
    https://docs.scipy.org/doc/scipy-0.18.1/reference/\
    generated/scipy.optimize.minimize.html

    This function is designed to be used with the `scipy.optimize.minimize`_

    The function first argument is the variable to be used in the
    minimization procedure.

    The sector displacement must be a flat array compatible with the
    dimensions of the input image and sectors shape (see parameters section
    below for more details).


    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html


    Parameters
    ----------

    sector_displacement_1d : ndarray_
        Array of displacements to apply to each sector. The dimensions are:
        sector_displacement_2d
        [ x (0) or y (1) displacement, i index of sector, j index of sector ].
        The shape of the sector displacements must be compatible with the
        input image and the block shape.
        The shape should be (2, mx, my) where mx and my are the numbers of
        sectors in the x and the y dimension.

    template_image : ndarray_  (ndim=2)
        Input image array (nx by ny pixels) where the sector displacement
        is applied.

    reference_image : ndarray_ (ndim=2)
        Image array to be used as reference (nx by ny pixels).

    blocks_shape : ndarray_ (ndim=2)
        Number of sectors in each dimension (x and y).
        blocks_shape.shape = (mx,my)

    smooth_gain : float
        Smoothness constrain gain


    Returns
    -------

    penalty : float
        Value of the cost function

    """

    sector_displacement_2d = sector_displacement_1d.reshape(
        *((2,) + tuple(blocks_shape)))

    residuals, smoothness_penalty = _cost_function(
        sector_displacement_2d, template_image,
        input_image, mask, smooth_gain)
    if debug:
        print()
        print("residuals", residuals)
        print("smoothness_penalty", smoothness_penalty)
    return residuals + smoothness_penalty

# TODO: input parameter check
# TODO: add keywords for minimization options
# TODO: add masked arrays support! 
def vet(input_images,
        factors=[32, 16, 4, 2, 1],
        smooth_gain=100,
        first_guess=None,
        intermediate_steps=False,
        verbose=True,
        indexing='xy'):
    '''
    Variational Echo Tracking Algorithm presented in
    `Laroche and Zawadzki (1995)`_  and used in the McGill Algorithm for
    Prediction by Lagrangian Extrapolation (MAPLE) described in
    `Germann and Zawadzki (2002)`_.

    .. _`Laroche and Zawadzki (1995)`:\
        http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

    .. _`Germann and Zawadzki (2002)`:\
        http://dx.doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2

    This algorithm computes the displacement field between two images
    ( the input_image with respect to the template image).
    The displacement is sought by minimizing sum of the residuals of the
    squared differences of the images pixels and the contribution of a
    smoothness constrain.

    In order to find the minimum an scaling guess procedure is applied,
    from larger scales
    to a finer scale. This reduces the changes that the minimization procedure
    converges to a local minima. The scaling guess is defined by the scaling
    factors (see **factors** keyword).

    The smoothness of the returned displacement field is controlled by the
    smoothness constrain gain (**smooth_gain** keyword).

    If a first guess is not given, zero displacements are used as first guess.


    Parameters
    ----------

    input_images : ndarray_
        Input images, sequence of 2D arrays, or 3D arrays.
        The first dimension represents the template_image and the input_image.
        The template_image (first element in first dimensions) denotes the
        reference image used to obtain the displacement (2D array).
        The second is the target image.
        The expected dimensions are (2,ny,nx).
        Be aware the the 2D images dimensions correspond to (y, x).

    factors : list or array, optional
        If dimension is 1, the same factors will be used both image dimensions
        (x and y). If is 2D, the each row determines the factors of the
        corresponding dimensions.
        The factors denotes the number of sectors in each dimension used in
        each scaling procedure.

    smooth_gain : float, optional
        Smooth gain factor

    first_guess : ndarray, optional_
        If first_guess is not present zeros are used as first guess.
        The shape should be compatible with the input image.
        That is (2,nx,ny).

    intermediate_steps : bool, optional
        If True, also return a list with the first guesses obtained during the
        scaling procedure. False, by default.


    verbose : bool, optional
        Verbosity enabled if True (default). 

    indexing : {‘xy’, ‘ij’}, optional
        Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output.
        If ‘xy’ is selected, the displacement field dimensions correspond
        to (x,y). Otherwise, the displacements corresponds to indexes of the
        2D images (i,j).
        
    
    Returns
    -------

    displacementField : ndarray_
        Displacement Field (2D array representing the transformation) that
        warps the template image into the input image.

    References
    ----------

    Laroche, S., and I. Zawadzki, 1995:
    Retrievals of horizontal winds from single-Doppler clear-air data by
    methods of cross-correlation and variational analysis.
    J. Atmos. Oceanic Technol., 12, 721–738.
    doi: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

    Germann, U. and I. Zawadzki, 2002:
    Scale-Dependence of the Predictability of Precipitation from Continental
    Radar Images.  Part I: Description of the Methodology.
    Mon. Wea. Rev., 130, 2859–2873,
    doi: 10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2.

    '''

    print("Running VET algorithm")
    
    if indexing not in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
            
    if verbose:
        def debug_print(*args, **kwargs):
            print(*args, **kwargs)
    else:
        def debug_print(*args, **kwargs):
            pass

    input_images = numpy.array(input_images, dtype=numpy.float64)

    if (input_images.ndim != 3 or input_images.shape[0] != 2):
        raise ValueError("input_images dimension mismatch.\n" +
                         "input_images.shape: %s\n" % str(input_images.shape)+
                         "(2, x, y ) dimensions expected.")


    # Get mask
    if isinstance(input_images , MaskedArray):
        mask = numpy.ma.getmaskarray(input_images)
    else:
        # Mask invalid data 
        input_images = numpy.ma.masked_invalid(input_images)
        mask = numpy.ma.getmaskarray(input_images)
    
    input_images[mask] = 0  # Remove any Nan from the raw data
    
    # Create a 2D mask with the rigth data type for _vet
    mask = numpy.asarray(numpy.any(mask, axis=0), dtype='int8') 
    
    template_image = numpy.asarray(input_images.data[0, :], dtype=numpy.float64)
    input_image = numpy.asarray(input_images.data[1, :], dtype=numpy.float64)


    # Check that the factors divide the domain
    factors = numpy.asarray(factors, dtype=numpy.int)

    if factors.ndim == 1 and template_image.shape[0] == template_image.shape[1]:

        new_factors = (numpy.zeros((2,) + factors.shape, dtype=numpy.int)
                       + factors.reshape((1, factors.shape[0]))
                       )
        factors = new_factors

    if factors.ndim != 2:
        if factors.shape[0] != 2:
            raise ValueError("Incorrect factors dimensions.\n",
                             "factor should be a 1D if the input images"
                             + " are square or 2D array otherwise")

    # Check that the factors divide the domain
    for i in range(factors.shape[1]):
        if (template_image.shape[0] % factors[0, i]) > 0:
            raise ValueError(
                "The factor %d does not divide x dimension" % factors[0, i])

        if (template_image.shape[1] % factors[1, i]) > 0:
            raise ValueError(
                "The factor %d does not divide y dimension" % factors[1, i])

    # Sort factors in descending order
    factors[0, :].sort()
    factors[1, :].sort()

    # Prepare first guest
    first_guess_shape = (2, int(factors[0, 0]), int(factors[1, 0]))

    if first_guess is None:
        initial_guess = numpy.zeros(first_guess_shape, order='C')
    else:
        if first_guess.shape != first_guess_shape:
            raise ValueError("The shape of the initial guess do not match " + 
                             "the factors coarse resolution")
        else:
            initial_guess = first_guess

    first_guess = None

    _template_image = numpy.asarray(template_image, dtype='float64')
    _input_image = numpy.asarray(input_image, dtype='float64')

    if intermediate_steps:
        scaling_guesses = list()
        
    for i in range(factors.shape[1]):
        # Minimize for each sector size

        sector_shape = (round_int(template_image.shape[0] / factors[0, i]),
                        round_int(template_image.shape[1] / factors[1, i]))

        debug_print("\nNumber of sectors: %s" % str(factors[:, i]))
        debug_print("Sector Shape:", sector_shape)

        if first_guess is None:
            first_guess = initial_guess
        else:

            first_guess = zoom(first_guess,
                               (1,
                                factors[0, i] / factors[0, i - 1],
                                factors[1, i] / factors[1, i - 1]),
                               order=1, mode='nearest')

        debug_print("Minimizing")

        result = minimize(cost_function,
                          first_guess.flatten(),
                          args=(_template_image, _input_image,
                                factors[:, i], mask,
                                smooth_gain),
                          method='CG',
                          options={'eps': 0.1, 'gtol': 0.1,
                                   'maxiter': 20, 'disp': True}
                          )

        first_guess = result.x.reshape(*(first_guess.shape))

        if verbose:
            cost_function(result.x,
                          _template_image, _input_image,
                          factors[:, i], mask,
                          smooth_gain,
                          debug=True)

        if first_guess is None:
            if indexing == 'xy':
                scaling_guesses.append(first_guess[::-1,:,:])
            else:
                scaling_guesses.append(first_guess)

    first_guess = zoom(first_guess,
                       (1,
                        _template_image.shape[0] / first_guess.shape[1],
                        _template_image.shape[1] / first_guess.shape[2]),
                       order=1, mode='nearest')
    
    if indexing == 'xy':
        first_guess = first_guess[::-1,:,:]
        
    if intermediate_steps:
        return first_guess, scaling_guesses

    return first_guess
