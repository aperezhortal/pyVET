# -*- coding: utf-8 -*-

# For python 3 portability
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from scipy.misc.common import face  # @UnresolvedImport
from scipy.misc.pilutil import imresize  # @UnresolvedImport

from matplotlib import pyplot
import numpy

from pyVET.vet import morph


# Convert from RGB to grey scale
def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# Image size to use in the example
image_size = 16 * 4 * 2 * 2

# Get only one color component of the racoon face
reference = numpy.array(
                numpy.squeeze(face()[:image_size,:image_size,0]),
                dtype=numpy.float64)

reference = numpy.array(rgb2gray(imresize(face(), (image_size, image_size))))

# Set the grid values (x,y)
positions = -image_size * 0.5 + numpy.arange(image_size)
positions /= image_size
x, y = numpy.meshgrid(positions,
                      positions)

# Create a reference displacement to be applied to the reference image
# A rotor displacement is applied
displacement = numpy.zeros((2, image_size, image_size))
displacement[0, :] = y
displacement[1, :] = -x + y
displacement *= 50
#displacement[:, :10] = 0
#displacement[:, -10:] = 0


morphedImage, mask = morph(reference, displacement)

# xx, yy = numpy.meshgrid(numpy.arange(image_size),
#                         numpy.arange(image_size))
#
# tform = displacement + numpy.asarray([xx,yy])
# morphedImage = warp(reference, tform)

# Show the displacement field
fig = pyplot.figure(0)
step = 16
pyplot.quiver(x[::step, ::step], y[::step, ::step],
              displacement[0,::step, ::step],
              displacement[1, ::step, ::step], scale=10)
pyplot.title("Displacement field")


# Show the reference Image and the morphed
fig = pyplot.figure(1)
pyplot.subplot(131, aspect='equal')
pyplot.imshow(reference, cmap=pyplot.get_cmap('gray'))
pyplot.title('Reference Image')

pyplot.subplot(132, aspect='equal')

pyplot.imshow(morphedImage, cmap=pyplot.get_cmap('gray'))
pyplot.title('Morphed image')

pyplot.subplot(133, aspect='equal')
morphed_image, mask = morph(morphedImage, -displacement)
pyplot.imshow(morphed_image,
              cmap=pyplot.get_cmap('gray'))
pyplot.title('Inverse Morphed image')

pyplot.show()


print("Done")
