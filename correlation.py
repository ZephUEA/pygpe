try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

def cross2D( tup1:tuple[float,float], tup2:tuple[float,float] ):
    return tup1[0] * tup2[1] - tup1[1] * tup2[0]

def createCircles( shape:tuple[int,int], stepsize:int=1, threshold:int=None) -> cp.ndarray:
    '''
    Returns a normalised cp.ndarray of circles of increasing size, we want to introduce a step size, as we dont care if this at pixel resolution
    '''
    def circle( i:np.ndarray, j:np.ndarray, k:np.ndarray, stepsize:int=1 ) -> np.ndarray:
        imod = np.concatenate( ( i[ :i.shape[0] // 2, :, : ], i[ i.shape[0] // 2:, :, : ] - i.shape[0] ), axis=0 )
        jmod = np.concatenate( ( j[ :, :j.shape[1] // 2, : ], j[ :, j.shape[1] // 2:, : ] - j.shape[1] ), axis=1 )
        d = np.sqrt( imod ** 2 + jmod ** 2 ) 
        smallCircle = d < stepsize * k + 1
        largeCircle = d >= stepsize * k
        ring = smallCircle & largeCircle
        summation = np.sum( ring, axis=(0,1) )
        return ring / summation
    if threshold is None:
        threshold = min(shape[0],shape[1])
    array = np.fromfunction( circle, ( shape[0], shape[1], min( min( shape[0], shape[1] ) // ( 2 * stepsize ), threshold ) ), stepsize=stepsize )

    return cp.array( array )

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

def transverseMagnetisation( psi, frame=None ):
    if frame is not None:
        plusComp = cp.array( psi["psi_plus"][:,:,frame] )
        zeroComp = cp.array( psi["psi_zero"][:,:,frame] )
        minusComp = cp.array( psi["psi_minus"][:,:,frame] )
    else:
        plusComp = cp.array( psi["psi_plus"][()] )
        zeroComp = cp.array( psi["psi_zero"][()] )
        minusComp = cp.array( psi["psi_minus"][()] )
    # The division by 2 has been moved to the squareing term, so that i dont introduce float errors by division by root 2 of the components
    fx = ( np.conjugate( plusComp + minusComp ) * zeroComp + np.conjugate( zeroComp ) * ( plusComp + minusComp )).real
    fy = ( 1j * ( np.conjugate(minusComp - plusComp ) * zeroComp + np.conjugate( zeroComp ) * ( plusComp - minusComp ) )).real
    return ( np.sqrt( ( abs(fx)**2 + abs(fy)**2 ) / 2 ), np.atan( fy/fx ) )

def gradient( psi, gridSpacing, frame=None ):
    if frame is not None:
        density = abs( cp.array( psi["psi_plus"][:,:,frame] ) ) ** 2 + abs( cp.array( psi["psi_zero"][:,:,frame] ) ) ** 2 + abs( cp.array( psi["psi_minus"][:,:,frame] ) ) ** 2
    else:
        density = abs( cp.array( psi["psi_plus"][()] ) ) ** 2 + abs( cp.array( psi["psi_zero"][()] ) ) ** 2 + abs( cp.array( psi["psi_minus"][()] ) ) ** 2
    gradient = np.gradient(density, gridSpacing, axis=(0,1))
    return gradient

def componentDensity( psi, component ):
    match component.lower():
        case 'psi_plus':
            waveFunc = psi['psi_plus']
        case 'psi_zero':
            waveFunc = psi['psi_zero']
        case 'psi_minus':
            waveFunc = psi['psi_minus']
    
    return np.sum( abs(waveFunc[()]) ** 2, axis=(0,1) )


def bestFitLine( xs: cp.ndarray, ys:cp.ndarray ) -> tuple[float,float]:
    b, m = cp.polynomial.polynomial.polyfit( xs, ys, 1 )
    return ( b, m )

def r2( xs: cp.ndarray, ys: cp.ndarray ) -> float:
    coeffs = cp.polynomial.polynomial.polyfit( xs, ys, 1 )
    p = cp.polynomial.polynomial.Polynomial( coeffs )
    yhat = p( xs )
    ybar = cp.sum( ys ) / len( ys )
    ssreg = cp.sum( ( yhat - ybar ) ** 2 )
    sstot = cp.sum( ( ys - ybar ) ** 2 )
    return ssreg / sstot

def bestFitCurveError( function, xs, ys, deltaYs=None, estimates=None, bounds=None ):
    if bounds:
        fit,cov = curve_fit( function, xs,ys, bounds=bounds )
    else:
        fit,cov = curve_fit( function, xs,ys, sigma=deltaYs, absolute_sigma=True, p0=estimates )
    error = np.sqrt( np.diag( cov ) )
    # var1 = fit[0]
    # var2 = fit[1]
    # errVar1 = np.sqrt(cov[0][0])
    # errVar2 = np.sqrt(cov[1][1])
    return [ i for i in zip(fit,error)]

def power_law_regression_logspace(x, y, y_err, method='weighted_ols'):
    """
    Fit power law y = b * x^m using log-space transformation
    
    Parameters:
    -----------
    x : array-like
        Independent variable (must be positive)
    y : array-like  
        Dependent variable (must be positive)
    y_err : array-like
        Symmetric errors on y in linear space
    method : str
        'weighted_ols' (default) or 'simple_ols'
    
    Returns:
    --------
    dict with results:
        - a, b: fitted parameters for y = a * x^b
        - a_err, b_err: uncertainties on parameters
        - r_squared: coefficient of determination
        - residuals: residuals in log space
    """
   
    x, y, y_err = np.asarray(x), np.asarray(y), np.asarray(y_err)
    
    # Check for positive values (required for log transform)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("All x and y values must be positive for log transformation")
    
    log_x = np.log(x)
    log_y = np.log(y)
    
    # For y with error ±σ_y, log error ≈ σ_y/y for small relative errors
    relative_err = y_err / y
    # log_y_err = relative_err  # This is the approximation σ_log ≈ σ_linear/y
    
    # For better accuracy with larger errors:
    log_y_err = (np.log(1 + relative_err) + np.log(1/(1 - relative_err))) / 2
    
    if method == 'weighted_ols':
        # Weighted least squares in log space
        weights = 1.0 / log_y_err**2
        
        # Calculate weighted regression coefficients
        w_sum = np.sum(weights)
        w_x = np.sum(weights * log_x)
        w_y = np.sum(weights * log_y)
        w_xx = np.sum(weights * log_x * log_x)
        w_xy = np.sum(weights * log_x * log_y)

        m = (w_sum * w_xy - w_x * w_y) / (w_sum * w_xx - w_x**2)
        log_b = (w_y - m * w_x) / w_sum
        residuals_log = log_y - (log_b + m * log_x)
        delta = w_sum * w_xx - w_x**2

        var_log_b = w_xx / delta
        var_m = w_sum / delta
        cov_log_b_m = -w_x / delta
        log_b_err = np.sqrt(var_log_b)
        m_err = np.sqrt(var_m)

        cov_matrix = np.array([[var_log_b, cov_log_b_m],[cov_log_b_m, var_m]])
        print( np.linalg.cond( cov_matrix ) )
        
    else:
        # Simple linear regression (unweighted)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        m = slope
        log_b = intercept
        m_err = std_err
        n = len(x)
        residuals_log = log_y - (log_b + m * log_x)
        mse = np.sum(residuals_log**2) / (n - 2)
        x_mean = np.mean(log_x)
        sxx = np.sum((log_x - x_mean)**2)
        log_b_err = np.sqrt(mse * (1/n + x_mean**2/sxx))
    
    b = np.exp(log_b)
    b_err = b * log_b_err
    log_y_mean = np.mean(log_y)
    ss_res = np.sum(residuals_log**2)
    ss_tot = np.sum((log_y - log_y_mean)**2)
    r_squared = 1 - ss_res/ss_tot
    
    return {
        'b': b,
        'm': m, 
        'b_err': b_err,
        'm_err': m_err,
        'r_squared': r_squared,
        'residuals_log': residuals_log,
        'log_b': log_b,
        'log_b_err': log_b_err
    }


if __name__ == '__main__':
    print( createCircles((50,50),stepsize=1).shape)
    print( np.linspace(0,10,11))