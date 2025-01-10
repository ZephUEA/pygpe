try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp

def cross2D( tup1:tuple, tup2:tuple ):
    return tup1[0] * tup2[1] - tup1[1] * tup2[0]

def createCircle( shape:tuple[int,int], distance:float, center:tuple[int,int]=(0,0) ) -> cp.ndarray:
    
    def modulo( n:int, m:int ) -> int:
        '''Returns a modulo function centered around 0'''
        modulus = n % m
        if modulus > m // 2:
            return modulus - m
        return modulus

    array = cp.zeros( shape )
    centerx = center[0]
    centery = center[1]
    count = 0
    for i in range( shape[0] ):
        for j in range( shape[1] ):
            d = cp.sqrt( modulo( centerx - i, shape[0] ) ** 2 + modulo( centery - j, shape[1] ) ** 2 )
            if distance <= d and d < distance + 1:
                array[i,j] = 1
                count += 1
    return array / count

def radialAverage( component:cp.ndarray, scalars:dict ) -> tuple[cp.ndarray]:
    nx = scalars["nx"]
    ny = scalars["ny"]
    dx = scalars["dx"] # Currently only works if dx = dy i.e. a square grid
    shape = ( nx, ny )
    size = min( nx, ny ) // 2 # With periodic BCs we dont want to start counting points closer to the center again
    averageArray = cp.zeros( ( size, component.shape[ 2 ] ) )
    for distance in range( size ):
        circle = createCircle( shape, distance )
        averageArray[distance,:] = cp.sum( circle[:,:,None] * component, axis=(0,1) )
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
   