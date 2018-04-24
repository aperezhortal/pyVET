# -*- coding: utf-8 -*-

# For python 3 portability
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from scipy.misc.common import face  # @UnresolvedImport
from scipy.misc.pilutil import imresize # @UnresolvedImport

import numpy

import matplotlib.pyplot as plt
from pyVET.VET import VET
from pyVET._VET import _morph, _costFunction


# Convert from RGB to grey scale
def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# Image size to use in the example
imageSize = 16 * 4 * 2 *2

# Get only one color component of the racoon face
reference = numpy.array(numpy.squeeze(face()[:imageSize, :imageSize, 0]), dtype=numpy.float64)
reference = numpy.array(rgb2gray(imresize(face(), (imageSize, imageSize))))

# Set the grid values (x,y) 
positions = -imageSize * 0.5 + numpy.arange(imageSize)
positions /= imageSize
x, y = numpy.meshgrid(positions,
                       positions)

# Create a reference displacement to be applied to the reference image
# A rotor displacement is applied
displacement = numpy.zeros((2, imageSize , imageSize))    
displacement[0, :] = y
displacement[1, :] = -x + y
displacement *= 50
#displacement[:, :10] = 0
#displacement[:, -10:] = 0


morphedImage = _morph(reference, displacement)

# Show the displacement field
fig = plt.figure(0)
step = 16
plt.quiver(x[::step, ::step], y[::step, ::step], displacement[0, ::step, ::step], displacement[1, ::step, ::step], scale=10)
plt.title("Displacement field")


factors = [  8,16, 32]

newDisplacement = VET( _morph( reference, displacement ), reference , factors , verbose=True,smoothGain=100)*-1

print(_costFunction(displacement,reference,morphedImage,100))
print(_costFunction(newDisplacement,reference,morphedImage,100))
plt.subplot( 121, aspect = 'equal' )
step = 16
plt.quiver( x[::step,::step], y[::step,::step], displacement[0,::step,::step], displacement[1,::step,::step], scale = 10 )
plt.subplot( 122, aspect = 'equal' )
plt.quiver( x[::step,::step], y[::step,::step], newDisplacement[0,::step,::step], newDisplacement[1,::step,::step], scale = 10 )

 
plt.show()

print("Done")