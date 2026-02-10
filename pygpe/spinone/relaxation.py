import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def rollWithZeros(arr, shift, axis=None):
    '''For use with the boundary conditions of 0 at all edges. '''
    result = np.roll(arr, shift, axis=axis)
    
    if axis is None:
        # Flat roll
        arr_flat = result.ravel()
        if shift > 0:
            arr_flat[:shift] = 0
        elif shift < 0:
            arr_flat[shift:] = 0
        return arr_flat.reshape(result.shape)
    else:
        # Roll along specific axis
        if shift > 0:
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(0, shift)
            result[tuple(slices)] = 0
        elif shift < 0:
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(shift, None)
            result[tuple(slices)] = 0
    
    return result

class Spinor():
    def __init__(self, psiPlus, psiZero, psiMinus ):
        self.plus = psiPlus
        self.zero = psiZero
        self.minus = psiMinus
    
    def __getitem__(self, index ):
        match index:
            case 1:
                return self.plus
            case 0:
                return self.zero
            case -1:
                return self.minus
            case _:
                raise IndexError('Spinor only has 1,0,-1 components')
    
    def __add__( self, other ):
        return Spinor( self.plus + other[1] , self.zero + other[0] , self.minus + other[-1])
    
    def __sub__( self, other ):
        return Spinor( self.plus - other[1] , self.zero - other[0] , self.minus - other[-1] )
    
    def __mul__(self, scalar):
        return Spinor( self.plus * scalar, self.zero*scalar, self.minus*scalar)
    
    def __rmul__(self, scalar):
        return self * scalar
    
    def __eq__(self,other):
        return ( np.array_equal( self[1], other[1] ) and np.array_equal( self[0], other[0] )  and np.array_equal( self[-1], other[-1] ) )
    
    def isClose( self, other, rtol=1e-05, atol=1e-08 ):
        return ( np.isclose(self[1], other[1],rtol,atol) and np.isclose(self[0], other[0],rtol,atol) and np.isclose(self[-1], other[-1],rtol,atol) )
    
    def __abs__(self):
        return np.sum( abs( self.plus )**2 + abs( self.zero )**2  + abs( self.minus ) ** 2 )
    
    
    def mag(self):
        return np.sum(  abs( self.plus )**2 - abs( self.minus ) ** 2 )
    
    def zeeman(self):
        return np.sum(  abs( self.plus )**2 + abs( self.minus ) ** 2 )
    

class SpinorBECGroundState2D():
    
    def __init__(self, grid, params,  psi ):
        """
        Parameters:
        -----------
        grid : 2D Grid object
        params : dict with 'c0', 'c2', 'trap', 'dt'
        psi : Spinor Object
        """
        self.grid                             = grid
        self.params:dict                      = params
        self.dx:float                         = grid.grid_spacing_x
        self.dy:float                         = grid.grid_spacing_y
        self.dt:float                         = params['dit'] if 'dit' in params.keys() else params['dt']
        self.waveFunctions: list[Spinor]      = [ psi ] # This will be the actual results over time
        self.trialWavefunctions: list[Spinor] = [] # This is a helper list
        self.tolerance: float                 = 1e-12
        
        # Stabilization parameters (tune these!)
        self.alpha1:float       = 0
        self.alpha0:float       = 0
        self.alpha_minus1:float = 0
        self.gradDebug = False
        self.intDebug = False
        

    def fullStep( self ):
        self.trialWavefunctions = [self.waveFunctions[-1]]
        self.nonlinearStep()
        while abs( self.trialWavefunctions[-2] - self.trialWavefunctions[-1] ) > self.tolerance**2:
            self.nonlinearStep()

        self.waveFunctions.append( self.trialWavefunctions[-1] )
        

    def nonlinearStep( self ):
        # Crank-Nicolson iteration
        psiPrevious = self.waveFunctions[-1]

        mu, lam = self.computeChemicalPotentials()
        vector = self.nonLinearCalc( mu, lam )
        # Now add the gradient and time dependent terms to the vector

        spaceTimeVector = Spinor( *[((1/self.dt - 1/(2*self.dx**2) - 1/(2*self.dy**2) ) * psiPrevious[j] + 
                         1/(4*self.dx**2) * ( rollWithZeros(psiPrevious[j], 1, axis=0 ) + rollWithZeros(psiPrevious[j], -1, axis=0 ) ) +  
                         1/(4*self.dy**2) * ( rollWithZeros(psiPrevious[j], 1, axis=1 ) + rollWithZeros(psiPrevious[j], -1, axis=1 ) )
                         ) for j in [1, 0, -1]] )

        if self.intDebug:
            spaceTimeVector = Spinor(*[ 1/self.dt * psiPrevious[j] for j in [1,0,-1]])

        constantPlus = vector[1] + spaceTimeVector[1]
        constantZero = vector[0] + spaceTimeVector[0]
        constantMinus = vector[-1] + spaceTimeVector[-1]


        nx = self.grid.shape[0]
        ny = self.grid.shape[1]
        shapeNum = nx * ny
        nyDiag = [-1/(4*self.dy**2)]*nx*(ny-1)
        nxDiag = ([-1/(4*self.dx**2)]*(nx-1) + [0]) * ny

        matrixPlus = diags( [ [self.alpha1 + 1/(self.dt) +1/(2*self.dx**2) + 1/(2*self.dy**2)]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr' )
        if self.intDebug:
            matrixPlus = diags([ [self.alpha1 + 1/(self.dt) ]*shapeNum, ], 
                             [0], shape = (shapeNum,shapeNum), format='csr')
        
        newPlus = spsolve( matrixPlus, constantPlus.flatten() ).reshape(self.grid.shape)

        matrixZero = diags( [ [self.alpha0 + 1/(self.dt) +1/(2*self.dx**2) + 1/(2*self.dy**2)]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr'  )
        if self.intDebug:
            matrixZero = diags([ [self.alpha0 + 1/(self.dt) ]*shapeNum, ], 
                             [0], shape = (shapeNum,shapeNum), format='csr')
        newZero = spsolve( matrixZero, constantZero.flatten() ).reshape(self.grid.shape)

        matrixMinus = diags( [ [self.alpha_minus1 + 1/(self.dt) +1/(2*self.dx**2) + 1/(2*self.dy**2)]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr'  )
        if self.intDebug:
            matrixMinus = diags([ [self.alpha_minus1 + 1/(self.dt) ]*shapeNum, ], 
                             [0], shape = (shapeNum,shapeNum), format='csr')
        newMinus = spsolve( matrixMinus, constantMinus.flatten() ).reshape(self.grid.shape)

        self.trialWavefunctions.append( Spinor( newPlus, newZero, newMinus ) )


    def nonLinearCalc( self, mu, lam ):
        psiPrevious = self.waveFunctions[-1]
        psiCurrent = self.trialWavefunctions[-1]
        psiHalf = Spinor( *[ 0.5*(psiPrevious[j] + psiCurrent[j]) for j in [1, 0, -1]] )
        psiSumTerm = sumTerm( psiCurrent, psiPrevious )
        psiDiffTerm = differenceTerm( psiCurrent, psiPrevious )
        psiSpinIndTerm = spinIndependentTerm( psiCurrent, psiPrevious )

        psiPlus = (
            self.alpha1 * psiCurrent[1] 
            - ( self.params['c0'] - self.params['c2'])/2 * psiDiffTerm[1] * psiHalf[1] 
            - (self.params['c0'] + self.params['c2'])/2 * psiSumTerm[1] * psiHalf[1] 
            - self.params['trap'] * psiHalf[1] 
            - self.params['c2']/2 * np.conj(psiHalf[-1]) * (psiCurrent[0]**2 + psiPrevious[0]**2) 
            + (mu + lam)*psiHalf[1]
        )

        psiZero = (
            self.alpha0 * psiCurrent[0] 
            - (self.params['c0']/2) * psiSpinIndTerm[0] * psiHalf[0] 
            - (self.params['c0'] + self.params['c2'])/2 * psiSumTerm[0] * psiHalf[0] 
            - self.params['trap'] * psiHalf[0] 
            - self.params['c2'] * np.conj(psiHalf[0]) * (psiCurrent[1]*psiCurrent[-1] + psiPrevious[1]*psiPrevious[-1] ) 
            + mu * psiHalf[0]
        )

        psiMinus = (
            self.alpha_minus1 * psiCurrent[-1] 
            - (self.params['c0'] - self.params['c2'])/2 * psiDiffTerm[-1] * psiHalf[-1] 
            - (self.params['c0'] + self.params['c2'])/2 *psiSumTerm[-1] * psiHalf[-1] 
            - self.params['trap'] * psiHalf[-1] 
            - self.params['c2']/2 * np.conj(psiHalf[1]) * (psiCurrent[0]**2 + psiPrevious[0]**2) 
            + (mu - lam)*psiHalf[-1]
        )

        if self.gradDebug:
            return Spinor( self.alpha1 * psiCurrent[1] + (mu + lam)*psiHalf[1], 
                           self.alpha0 * psiCurrent[0] + mu * psiHalf[0], 
                           self.alpha_minus1 * psiCurrent[-1] + (mu - lam)*psiHalf[-1] )
        
        return Spinor( psiPlus, psiZero, psiMinus )

    def computeChemicalPotentials(self):
        """
        Compute μ and λ, The chemical and magnetic potentials
        These ensure mass and magnetization conservation
        """
        psiPrevious = self.waveFunctions[-1]
        psiCurrent = self.trialWavefunctions[-1]

        psiHalf = Spinor( *[0.5*(psiPrevious[j] + psiCurrent[j]) for j in [1, 0, -1]] )
        
        
        nHalf = abs(psiHalf)
        mHalf = psiHalf.mag()
        rHalf = psiHalf.zeeman()
        dHalf, fHalf = self.compute_D_F(psiCurrent, psiPrevious, psiHalf)
        
        denom = nHalf * rHalf - mHalf**2
        if denom == 0:
            return 0, 0
        
        mu = (rHalf * dHalf - mHalf * fHalf) / denom
        lam = (nHalf * fHalf - mHalf * dHalf) / denom
        
        return mu, lam
    
    def compute_D_F( self, psiCurrent, psiPrevious, psiHalf ):

        psiSumTerm = sumTerm( psiCurrent, psiPrevious )
        psiDiffTerm = differenceTerm( psiCurrent, psiPrevious )
        psiSpinIndTerm = spinIndependentTerm( psiCurrent, psiPrevious )


        posIndependentTermD = ( 
            self.params['c0']/2 * psiSpinIndTerm[0] * abs(psiHalf[0])**2 
            + (self.params['c0'] - self.params['c2'])/2 * ( psiDiffTerm[1] * abs(psiHalf[1])**2 + psiDiffTerm[-1] * abs(psiHalf[-1])**2 ) 
            + self.params['c2'] * ( psiHalf[-1]*np.conj(psiCurrent[0]**2 + psiPrevious[0]**2) * psiHalf[1] 
                                   + np.conj(psiHalf[0]**2) * (psiCurrent[-1]*psiCurrent[1] + psiPrevious[-1]*psiPrevious[1]) ).real 
            + (self.params['c0'] + self.params['c2'])/2 * ( psiSumTerm[1] * abs(psiHalf[1])**2 + psiSumTerm[-1] * abs(psiHalf[-1])**2 + psiSumTerm[0] * abs(psiHalf[0])**2 )
        )
        # What direction is correct to roll?
        rolledSpinorX = Spinor( rollWithZeros(psiHalf[1],1,axis=0), rollWithZeros(psiHalf[0],1,axis=0), rollWithZeros(psiHalf[-1],1,axis=0) )
        rolledSpinorY = Spinor( rollWithZeros(psiHalf[1],1,axis=1), rollWithZeros(psiHalf[0],1,axis=1), rollWithZeros(psiHalf[-1],1,axis=1) )


        rolls = Spinor( *[(1/(2*self.dx**2)) * ( rollWithZeros(psiHalf[j],1,axis=0) + rollWithZeros(psiHalf[j],-1,axis=0) - 2 * psiHalf[j]) * psiHalf[j]
                    + (1/(2*self.dy**2)) * ( rollWithZeros(psiHalf[j],1,axis=1) + rollWithZeros(psiHalf[j],-1,axis=1) - 2 * psiHalf[j] ) * psiHalf[j] for j in [1,0,-1] ] )

        boundary = boundaryTerm( psiHalf, self.dx, self.dy )

        gradTermD = ( (1/(2*self.dx**2))*( abs(rolledSpinorX[1]-psiHalf[1])**2 + abs(rolledSpinorX[0]-psiHalf[0])**2 + abs(rolledSpinorX[-1]-psiHalf[-1])**2  )
                     + (1/(2*self.dy**2))*( abs(rolledSpinorY[1]-psiHalf[1])**2 + abs(rolledSpinorY[0]-psiHalf[0])**2 + abs(rolledSpinorY[-1]-psiHalf[-1])**2) )
        
        boundaryD = -(boundary[1] + boundary[0] + boundary[-1])/2
        # gradTermD = -( rolls[1] + rolls[0] + rolls[-1] )/2
        trapTermD = self.params['trap'] * ( abs(psiHalf[1])**2 + abs(psiHalf[0])**2 + abs(psiHalf[-1])**2 )

        posIndependentTermF = ( 
            (self.params['c0'] - self.params['c2'])/2 * ( psiDiffTerm[1] * abs(psiHalf[1])**2 - psiDiffTerm[-1] * abs(psiHalf[-1])**2 ) 
            + (self.params['c0'] + self.params['c2'])/2 * ( psiSumTerm[1] * abs(psiHalf[1])**2 -  psiSumTerm[-1] * abs(psiHalf[-1])**2)
        )

        gradTermF =  ( (1/(2*self.dx**2))*( abs(rolledSpinorX[1]-psiHalf[1])**2 - abs(rolledSpinorX[-1]-psiHalf[-1])**2  ) + 
                      (1/(2*self.dy**2))*( abs(rolledSpinorY[1]-psiHalf[1])**2 - abs(rolledSpinorY[-1]-psiHalf[-1])**2))
        
        boundaryF = -(boundary[1] - boundary[-1])/2
        
        # gradTermF = -( rolls[1] - rolls[-1] )/2
        trapTermF = self.params['trap'] * ( abs(psiHalf[1])**2 - abs(psiHalf[-1])**2 )
        
        if self.gradDebug:
            return ( np.sum( gradTermD ) + boundaryD, np.sum( gradTermF ) + boundaryF )
        
        if self.intDebug:
            return ( np.sum( posIndependentTermD ), np.sum( posIndependentTermF ) )

        return ( np.sum( gradTermD + trapTermD + posIndependentTermD ), np.sum( gradTermF + trapTermF + posIndependentTermF ) )



def sumTerm( psiCurrent, psiPrevious ):
    newPlus = abs(psiCurrent[1])**2 +abs(psiPrevious[1])**2 + abs(psiCurrent[0])**2 +abs(psiPrevious[0])**2
    newZero = abs(psiCurrent[1])**2 +abs(psiPrevious[1])**2 + abs(psiCurrent[-1])**2 +abs(psiPrevious[-1])**2
    newMinus = abs(psiCurrent[0])**2 +abs(psiPrevious[0])**2 + abs(psiCurrent[-1])**2 +abs(psiPrevious[-1])**2
    return Spinor( newPlus, newZero, newMinus )

def differenceTerm( psiCurrent, psiPrevious ):
    newPlus = abs(psiCurrent[-1])**2 + abs(psiPrevious[-1])**2
    newZero = 0
    newMinus = abs(psiCurrent[1])**2 + abs(psiPrevious[1])**2
    return Spinor( newPlus, newZero, newMinus )

def spinIndependentTerm( psiCurrent, psiPrevious ):
    newPlus = 0
    newZero = abs(psiCurrent[0])**2 + abs(psiPrevious[0])**2
    newMinus = 0
    return Spinor( newPlus, newZero, newMinus )

def boundaryTerm( psi, dx, dy ):
    # Integrate psi grad psi around the boundary.

    np.sum( np.conj(psi[1]) * (rollWithZeros( psi[1], 1 , axis=1 ) - psi[1] )[-1,:] )
           


    return Spinor(*[( 1j*np.sum(( np.conj(psi[j]) * (rollWithZeros( psi[j], 1 , axis=0 ) - psi[j] )/dx)[:,0]  )
         + 1j*np.sum(( np.conj(psi[j]) * (rollWithZeros( psi[j], 1 , axis=1 ) - psi[j] )/dy )[-1,:]  )
         + 1j*np.sum(( np.conj(psi[j]) * (-rollWithZeros( psi[j], -1 , axis=0 ) + psi[j] )/dx )[:,-1]  )
         + 1j*np.sum(( np.conj(psi[j]) * (-rollWithZeros( psi[j], -1 , axis=1 ) + psi[j] )/dy )[0,:]  )
        ) for j in [1,0,-1]])


if __name__ == '__main__':
    arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    print( rollWithZeros( arr, 1, axis=0))
    print( arr.shape )

    nx = 3
    ny = 3
    shapeNum = nx * ny
    nyDiag = [-1/(4)]*nx*(ny-1)
    nxDiag = ([-1/(4)]*(nx-1) + [0]) * ny

    matrixPlus = diags( [ [1]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum)).toarray()
    print( matrixPlus )