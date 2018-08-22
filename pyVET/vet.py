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
    return int(numpy.round(scalar))


def ceil_int(scalar):
    """
    Round number to nearest integer. Returns and integer value.
    """
    return int(numpy.ceil(scalar))


def get_padding(dimension_size, sectors):
    """
    Get the padding at each side of the one dimensions of the image
    so the new image dimensions are divided evenly in the
    number of *sectors* specified.

    Parameters
    ----------

    dimension_size : int
        Actual dimension size.

    sectors : int
        number of sectors over which the the image will be divided.

    Return
    ------

    pad_before , pad_after: int, int
        Padding at each side of the image for the corresponding dimension.
    """
    reminder = dimension_size % sectors

    if reminder != 0:
        pad = sectors - reminder
        pad_before = pad // 2
        if (pad % 2 == 0):
            pad_after = pad_before
        else:
            pad_after = pad_before + 1

        return pad_before, pad_after

    return (0, 0)


def morph(image, displacement, indexing='ij'):
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
        Displacement field to be applied (Warping). The first dimension
        corresponds to the coordinate to displace, depending on the indexing
        convention (see indexing parameter below).

        The dimensions are:
        displacement [ i/x (0) or j/y (1) ,
                      i index of pixel, j index of pixel ]



    indexing : {‘xy’, ‘ij’}, optional
        Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output.
        If ‘xy’ is selected, the displacement field dimensions correspond
        to (x,y). Otherwise, the displacements corresponds to indexes of the
        2D images (i,j).

    Returns
    -------

    morphed_image : ndarray (float64 ,ndim = 2)
        Morphed image

    mask : ndarray (int8 ,ndim = 2)
        Invalid values mask. Points outside the boundaries are masked.
        Values greater than 1, indicate masked values.
    """

    if indexing not in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    _image = numpy.asarray(image, dtype='float64', order='C')
    _displacement = numpy.asarray(displacement, dtype='float64', order='C')

    if indexing == 'xy':
        _displacement = _displacement[::-1, :, :]

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

    downsized_array : MaskedArray
        Downsized array with the invalid entries masked.

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


# TODO: add keywords for minimization options
# TODO: Add mask to padded values!!!
def vet(input_images,
        sectors=[[32, 16, 4, 2, 1], [32, 16, 4, 2, 1]],
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
    converges to a local minimum. The scaling guess is defined by the scaling
    sectors (see **sectors** keyword).

    The smoothness of the returned displacement field is controlled by the
    smoothness constrain gain (**smooth_gain** keyword).

    If a first guess is not given, zero displacements are used as first guess.

    .. _MaskedArray: https://docs.scipy.org/doc/numpy/reference/\
        maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------

    input_images : ndarray_ or MaskedArray
        Input images, sequence of 2D arrays, or 3D arrays.
        The first dimension represents the template_image and the input_image.
        The template_image (first element in first dimensions) denotes the
        reference image used to obtain the displacement (2D array).
        The second is the target image.
        The expected dimensions are (2,ny,nx).
        Be aware the the 2D images dimensions correspond to (y, x).

    sectors : list or array, optional
        The number of sectors for each dimension used in the scaling procedure.
        If dimension is 1, the same sectors will be used both image dimensions
        (x and y). If is 2D, the each row determines the sectors of the
        each dimension.

    smooth_gain : float, optional
        Smooth gain factor

    first_guess : ndarray_, optional_
        The shape of the first guess should have the same shape as the initial
        sectors shapes used in the scaling procedure.
        If first_guess is not present zeros are used as first guess.

        E.g.:
            If the first sector shape in the scaling procedure is (ni,nj), then
            the first_guess should have (2, ni, nj ) shape.

    intermediate_steps : bool, optional
        If True, also return a list with the first guesses obtained during the
        scaling procedure. False, by default.

    verbose : bool, optional
        Verbosity enabled if True (default).

    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        If 'xy' is selected, the displacement field dimensions correspond
        to (x,y).
        Otherwise, the displacements corresponds to indexes of the
        2D images (i,j).

        E.g.:
            If 'ij' is selected, the returned displacementField first dimension
            denote the displacement along the row (displacementField[0,:,:])
            and along the column (displacementField[1,:,:]).

            If 'xy' is selected, the returned displacementField first dimension
            denote the displacement along the x dimension
            (displacementField[0,:,:]) or along the y dimension
            (displacementField[1,:,:]).


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

    if indexing not in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    if verbose:
        def debug_print(*args, **kwargs):
            print(*args, **kwargs)
    else:
        def debug_print(*args, **kwargs):
            pass

    debug_print("Running VET algorithm")

    input_images = numpy.array(input_images, dtype=numpy.float64)

    if (input_images.ndim != 3 or input_images.shape[0] != 2):
        raise ValueError("input_images dimension mismatch.\n" +
                         "input_images.shape: %s\n" % str(input_images.shape) +
                         "(2, x, y ) dimensions expected.")

    # Get mask
    if isinstance(input_images, MaskedArray):
        mask = numpy.ma.getmaskarray(input_images)
    else:
        # Mask invalid data
        input_images = numpy.ma.masked_invalid(input_images)
        mask = numpy.ma.getmaskarray(input_images)

    input_images[mask] = 0  # Remove any Nan from the raw data

    # Create a 2D mask with the rigth data type for _vet
    mask = numpy.asarray(numpy.any(mask, axis=0), dtype='int8')

    template_image = numpy.asarray(
        input_images.data[0, :], dtype=numpy.float64)
    input_image = numpy.asarray(input_images.data[1, :], dtype=numpy.float64)

    # Check that the sectors divide the domain
    sectors = numpy.asarray(sectors, dtype=numpy.int)

    if sectors.ndim == 1:

        new_sectors = (numpy.zeros((2,) + sectors.shape, dtype=numpy.int)
                       + sectors.reshape((1, sectors.shape[0]))
                       )
        sectors = new_sectors
    elif sectors.ndim > 2 or sectors.ndim < 1:
        raise ValueError("Incorrect sectors dimensions.\n",
                         + "Only 1D or 2D arrays are supported to define"
                         + "the number of sectors used in"
                         + "the scaling procedure")

    # Sort sectors in descending order
    sectors[0, :].sort()
    sectors[1, :].sort()

    # Prepare first guest
    first_guess_shape = (2, int(sectors[0, 0]), int(sectors[1, 0]))

    if first_guess is None:
        first_guess = numpy.zeros(first_guess_shape, order='C')
    else:
        if first_guess.shape != first_guess_shape:
            raise ValueError("The shape of the initial guess do not match " +
                             "the sectors coarse resolution")
        else:
            first_guess = numpy.asarray(first_guess, order='C')

    template_image = numpy.asarray(template_image, dtype='float64')
    input_image = numpy.asarray(input_image, dtype='float64')

    scaling_guesses = list()

    previous_sectors_in_i = sectors[0, 0]
    previous_sectors_in_j = sectors[1, 0]

    for n, (sectors_in_i, sectors_in_j) in enumerate(
            zip(sectors[0, :], sectors[1, :])):

        # Minimize for each sector size
        pad_i = get_padding(template_image.shape[0], sectors_in_i)
        pad_j = get_padding(template_image.shape[1], sectors_in_j)

        if (pad_i != (0, 0)) or (pad_j != (0, 0)):

            _template_image = numpy.pad(template_image, (pad_i, pad_j), 'edge')
            _input_image = numpy.pad(input_image, (pad_i, pad_j), 'edge')

            if first_guess is None:
                first_guess = numpy.pad(first_guess, (pad_i, pad_j), 'edge')
        else:
            _template_image = template_image
            _input_image = input_image

        sector_shape = (_template_image.shape[0] // sectors_in_i,
                        _template_image.shape[1] // sectors_in_j)

        debug_print("\nNumber of sectors: %d,%d" %
                    (sectors_in_i, sectors_in_j))

        debug_print("Sector Shape:", sector_shape)

        if n > 0:
            first_guess = zoom(first_guess,
                               (1,
                                sectors_in_i / previous_sectors_in_i,
                                sectors_in_j / previous_sectors_in_j),
                               order=1, mode='nearest')

        debug_print("Minimizing")

        result = minimize(cost_function,
                          first_guess.flatten(),
                          args=(_template_image, _input_image,
                                (sectors_in_i, sectors_in_j),
                                mask,
                                smooth_gain),
                          method='CG',
                          options={'eps': 0.1, 'gtol': 0.1,
                                   'maxiter': 20, 'disp': True}
                          )

        first_guess = result.x.reshape(*first_guess.shape)

        if verbose:
            cost_function(result.x,
                          _template_image, _input_image,
                          (sectors_in_i, sectors_in_j),
                          mask,
                          smooth_gain,
                          debug=True)

        if indexing == 'xy':
            scaling_guesses.append(first_guess[::-1, :, :])
        else:
            scaling_guesses.append(first_guess)

        previous_sectors_in_i = sectors_in_i
        previous_sectors_in_j = sectors_in_j

    first_guess = zoom(first_guess,
                       (1,
                        _template_image.shape[0] / sectors_in_i,
                        _template_image.shape[1] / sectors_in_j),
                       order=1, mode='nearest')

    # Remove the extra padding if any

    ni = _template_image.shape[0]
    nj = _template_image.shape[1]

    first_guess = first_guess[:,
                  pad_i[0]:ni - pad_i[1],
                  pad_j[0]:nj - pad_j[1]]

    if indexing == 'xy':
        first_guess = first_guess[::-1, :, :]

    if intermediate_steps:
        return first_guess, scaling_guesses

    return first_guess
