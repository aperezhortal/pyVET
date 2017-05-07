    
from __future__ import division

import sys

import numpy
from scipy.ndimage.interpolation import zoom
from scipy.optimize._minimize import minimize
from skimage.util.shape import view_as_blocks

from pyVET.morphing import _costFunction, _morph


def roundInt( scalar ):
    
    return int( round( scalar ) )


class fatalError( Exception ):   
    """ Fatal error exception """   
    def __init__( self, message ):
        Exception.__init__( self, message )
        self.message = '\n' + message + '\n'
        
    def __repr__( self ):
        return ( self.message )    


def costFunction( sectorDisplacement, inputImage, referenceImage, blocksShape, smoothGain ):    
    
    sectorDisplacement2D = sectorDisplacement.reshape( *( ( 2, ) + blocksShape ) )
    
    return _costFunction( sectorDisplacement2D, inputImage, referenceImage, smoothGain )



def VET( inputImage, referenceImage, factors = [ 64, 16, 4, 2, 1 ], smoothGain = 10, firstGuess = None ):
    '''
    Variational Echo Tracking 
    
    Find the displacement vector of an image that best match a reference image using 
    an specific constrain (cost function) 
    
    TODO input parameter check
    
    Parameters
    ----------
    
    inputImage : numpy ndarray
        Input image 
    
    referenceImage : numpy ndarray
        Reference image used to obtain the displacement
        
    
        
    factors : list or array 
        If dimension is 1, the same factors will be used both imag dimensions (x and y)
        If is 2D, the each row determines the factors of the corresponding dimensions
        
    smoothGain : float
        Smooth gain factor
        
    firstGuess : ndarray
        If firstGuess is not preseent zeros are used as first guess
    
    
    
    '''
    
     
    inputImage = numpy.array( inputImage, dtype = numpy.float64 )
    referenceImage = numpy.array( referenceImage, dtype = numpy.float64 )
    
    factors = numpy.array( factors, dtype = numpy.int )
    
    if factors.ndim == 1 and inputImage.shape[0] == inputImage.shape[1]:
        newFactors = numpy.zeros( ( 2, ) + factors.shape ) + factors.reshape( ( 1, factors.shape[0] ) )
        factors = newFactors
        
    if factors.ndim != 2:
        if factors.shape[0] != 2:
            raise Exception( "The sectors array shape need to be (2,n) " )
        
    
    factors[0, :] = numpy.sort( factors[0, :] )
    #factors[0, :] = factors[0, :].copy()[::-1]   
    factors[1, :] = numpy.sort( factors[1, :] )
    #factors[1, :] = factors[1, :].copy()[::-1]
    #print factors
         
    for i in range( factors.shape[1] ):
        
        if ( inputImage.shape[0] % factors[0, i] ) > 0:
            raise Exception( "The factor %d does not divide x dimension" % factors[0, i] )
        
        if ( inputImage.shape[1] % factors[1, i] ) > 0:
            raise Exception( "The factor %d does not divide y dimension" % factors[1, i] )
    
    if firstGuess is None:
        firstGuess = numpy.zeros( ( 2, ) + inputImage.shape, order = 'C' )
    else:
        if firstGuess.shape != ( 2, ) + inputImage.shape:
            raise fatalError( "The shape of the initial guess do not match the Iput image dimensions" )


    for i in range( factors.shape[1] ):
        
        # Minimize for each sector size
        
        sectorShape = ( roundInt( inputImage.shape[0] / factors[0, i] ),
                        roundInt( inputImage.shape[1] / factors[1, i] ) )
        
        sectorFirstGuess = view_as_blocks( firstGuess,
                                           block_shape = ( 1, ) + sectorShape )
        
        print sectorFirstGuess.shape
        blocksShape = sectorFirstGuess.shape[1:3]
        print blocksShape
#         import cProfile, pstats, StringIO
#         pr = cProfile.Profile()
#         pr.enable()
        #initialGuess=sectorFirstGuess
        initialGuess = numpy.mean( numpy.mean( sectorFirstGuess,
                                               axis = 4, keepdims = True ),
                                               axis = 5, keepdims = True )
        
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





if __name__ == "__main__":
    
    from scipy.misc.common import face  # @UnresolvedImport
    print "Init"
    
    
    size = 16 * 4 *2 
    
    
    
    reference = numpy.array( numpy.squeeze( face()[:size, :size, 0] ), dtype = numpy.float64 )
    
    displacement = numpy.zeros( ( 2, size , size ) )

    
    positions = -size * 0.5 + numpy.arange( size )
    positions /= size
    
    x, y = numpy.meshgrid( positions,
                           positions )
        
    displacement[0, :] = y
    displacement[1, :] = -x + y
    displacement *= 4
    displacement[:,:10]=0
    displacement[:,-10:]=0
    
    import matplotlib.pyplot as plt
    
    plt.subplot( 121, aspect = 'equal' )
    plt.imshow(reference)
    plt.subplot( 122, aspect = 'equal' )
    plt.imshow(_morph( _morph( reference, displacement ), -displacement ))
    plt.show()
    factors = [ 4, 8,16, 32]
    
    newDisplacement = VET( _morph( reference, displacement ), reference , factors )*-1
    
    plt.subplot( 121, aspect = 'equal' )
    step = 16
    plt.quiver( x[::step,::step], y[::step,::step], displacement[0,::step,::step], displacement[1,::step,::step], scale = 10 )
    plt.subplot( 122, aspect = 'equal' )
    plt.quiver( x[::step,::step], y[::step,::step], newDisplacement[0,::step,::step], newDisplacement[1,::step,::step], scale = 10 )
    
     
    plt.show()
    
    print "Done"
