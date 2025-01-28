try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp

def cross2D( tup1:tuple[float,float], tup2:tuple[float,float] ):
    return tup1[0] * tup2[1] - tup1[1] * tup2[0]

def createCircles( shape:tuple[int,int] ):
    '''
    Returns a normalised cp.ndarray of circles of increasing size
    '''
    def circle( i:cp.ndarray, j:cp.ndarray, k:cp.ndarray ) -> cp.ndarray:
        i = cp.array( i ) # Have to coerce i,j,k from np.arrays to cp.arrays
        j = cp.array( j )
        k = cp.array( k )
        imod = cp.concatenate( ( i[ :i.shape[0] // 2, :, : ], i[ i.shape[0] // 2:, :, : ] - i.shape[0] ), axis=0 )
        jmod = cp.concatenate( ( j[ :, :j.shape[1] // 2, : ], j[ :, j.shape[1] // 2:, : ] - j.shape[1] ), axis=1 )
        d = cp.sqrt( imod ** 2 + jmod ** 2 ) 
        smallCircle = d < k +1
        largeCircle = d >= k
        ring = smallCircle & largeCircle
        summation = cp.sum( ring, axis=(0,1))
        return ring / summation

    array = cp.fromfunction( circle, ( shape[0], shape[1], min( shape[0], shape[1] ) // 2 ) )

    return array

def radialAverage( component:cp.ndarray, scalars:dict ) -> tuple[cp.ndarray, cp.ndarray]:
    nx = scalars["nx"]
    ny = scalars["ny"]
    dx = scalars["dx"] # Currently only works if dx = dy i.e. a square grid
    shape = ( nx, ny )
    size = min( nx, ny ) // 2
    circles = createCircles( shape )
    averageArray = cp.zeros( ( size, component.shape[2] ) )
    for i in range( component.shape[2] ):
        averageArray[:,i] = cp.sum( circles * component[:,:,i,None], axis=(0,1) )
    # averageArray = cp.sum( circles[:,:,:,None] * component[:,:,None,:], axis=(0,1) ) #Uses way too much memory
    xs = cp.linspace( 0, size * dx, size )
    return ( xs, averageArray )

def correlatorFM( psi:dict, scalars:dict ) -> cp.ndarray:
    nx = scalars["nx"]
    ny = scalars["ny"]
    plusComp = cp.array( psi['psi_plus'][()] )
    minusComp = cp.array( psi['psi_minus'][()] )
    magnetisation = abs( plusComp ) ** 2 - abs( minusComp ) ** 2
    magFFT = cp.fft.fft2( magnetisation, axes=(0,1) )
    autoCorrelation = cp.fft.ifft2( abs( magFFT ) ** 2, axes=(0,1) ).real
    return autoCorrelation / ( nx * ny )

def correlatorBA( psi:dict, scalars:dict ) -> cp.ndarray:
    nx = scalars["nx"]
    ny = scalars["ny"]
    plusComp = cp.array( psi['psi_plus'][()] )
    zeroComp = cp.array( psi["psi_zero"][()] )
    minusComp = cp.array( psi['psi_minus'][()] )
    magX = ( cp.conj(plusComp) * zeroComp + cp.conj(zeroComp) * ( plusComp + minusComp ) + cp.conj( minusComp ) * zeroComp ) / cp.sqrt(2)
    magY = 1j * ( -cp.conj(plusComp) * zeroComp + cp.conj(zeroComp) * ( plusComp - minusComp ) + cp.conj( minusComp ) * zeroComp) / cp.sqrt(2)
    magnetisation = abs( magX ) ** 2 + abs( magY ) ** 2
    magFFT = cp.fft.fft2( magnetisation, axes=(0,1) )
    autoCorrelation = cp.fft.ifft2( abs( magFFT ) ** 2, axes=(0,1) ).real
    return autoCorrelation / ( nx * ny )

        
def pseudoVorticity( psi:dict ) -> dict:
    plusComp = cp.array( psi["psi_plus"][()] )
    zeroComp = cp.array( psi["psi_zero"][()] )
    minusComp = cp.array( psi["psi_minus"][()] )
    plusVorticity = cross2D( cp.gradient( plusComp.real, axis=(0,1) ), cp.gradient( plusComp.imag, axis=(0,1) ) )
    zeroVorticity = cross2D( cp.gradient( zeroComp.real, axis=(0,1) ), cp.gradient( zeroComp.imag, axis=(0,1) ) )
    minusVorticity = cross2D( cp.gradient( minusComp.real, axis=(0,1) ), cp.gradient( minusComp.imag, axis=(0,1) ) )
    return { 'psi_plus':plusVorticity, 'psi_zero':zeroVorticity, 'psi_minus': minusVorticity }
   
def firstZero( arrIn:cp.ndarray ) -> int:
    for index, element in enumerate( arrIn ):
        if element * arrIn[ 0 ] <= 0:
            return index
    raise ValueError( 'Function has no Zeros' )

def bestFitLine( xs: cp.ndarray, ys:cp.ndarray ) -> tuple[float,float]:
    b, m = cp.polynomial.polynomial.polyfit( xs, ys, 1)
    return ( b, m )

if __name__ == '__main__':
    print( firstZero(cp.linspace(-10,10,100)) )