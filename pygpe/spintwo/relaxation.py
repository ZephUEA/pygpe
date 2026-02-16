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
    def __init__(self, psiPlus2, psiPlus1, psiZero, psiMinus1, psiMinus2 ):
        self.plus2 = psiPlus2
        self.plus1 = psiPlus1
        self.zero = psiZero
        self.minus1 = psiMinus1
        self.minus2 = psiMinus2
    
    def __getitem__(self, index ):
        match index:
            case 2:
                return self.plus2
            case 1:
                return self.plus1
            case 0:
                return self.zero
            case -1:
                return self.minus1
            case -2:
                return self.minus2
            case _:
                raise IndexError('Spinor only has 2,1,0,-1,-2 components')
    
    def __add__( self, other ):
        return Spinor( self[2] + other[2], self[1] + other[1] , self[0] + other[0] , self[-1] + other[-1], self[-2] + other[-2] )
    
    def __sub__( self, other ):
        return Spinor( self[2] - other[2], self[1] - other[1] , self[0] - other[0] , self[-1] - other[-1], self[-2] - other[-2] )
    
    def __mul__(self, scalar):
        return Spinor( self[2] * scalar, self[1] * scalar , self[0] * scalar , self[-1] * scalar, self[-2] * scalar )
    
    def __rmul__(self, scalar):
        return self * scalar
    
    def __eq__(self,other):
        return ( np.array_equal(self[2],other[2]) and np.array_equal( self[1], other[1] ) and np.array_equal( self[0], other[0] )  
                and np.array_equal( self[-1], other[-1] ) and np.array_equal(self[-2],other[-2]))
    
    def isClose( self, other, rtol=1e-05, atol=1e-08 ):
        return (np.isclose(self[2], other[2],rtol,atol) and np.isclose(self[1], other[1],rtol,atol) and np.isclose(self[0], other[0],rtol,atol) 
                and np.isclose(self[-1], other[-1],rtol,atol) and np.isclose(self[-2], other[-2],rtol,atol))
    
    def __abs__(self):
        return np.sqrt( np.sum( abs( self[2] )**2 +  abs( self[1] )**2 + abs( self[0] )**2  + abs( self[-1] )**2 + abs( self[-2] )**2  ) )



    def number(self):
        return np.sum( abs( self[2] )**2 +  abs( self[1] )**2 + abs( self[0] )**2  + abs( self[-1] )**2 + abs( self[-2] )**2  )
    
    
    def mag(self):
        return np.sum( 2 * ( abs( self[2] )**2 - abs( self[-2] )**2 ) +  abs( self[1] )**2 - abs( self[-1] )**2 )
    
    def zeeman(self):
        return np.sum( 4*(abs( self[2] )**2 + abs( self[-2] )**2) + abs( self[1] )**2 + abs( self[-1] )**2 )
    
    def localNumber( self ):
        return abs( self[2] )**2 + abs( self[1] )**2 + abs( self[0] )**2  + abs( self[-1] )**2 + abs( self[-2] )**2
    
    def localMag( self ):
        return 2 * ( abs( self[2] )**2 - abs( self[-2] )**2 ) +  abs( self[1] )**2 - abs( self[-1] )**2
    
    def localZeeman( self ):
        return 4*(abs( self[2] )**2 + abs( self[-2] )**2) + abs( self[1] )**2 + abs( self[-1] )**2

    def localSpinSinglet( self ):
        return ( 2 * self[2] * self[-2] - 2 * self[1] * self[-1] + self[0]**2 )/np.sqrt(5)
    
    def localMagPlus(self):
        return 2 * ( np.conj(self[2]) * self[1] + np.conj(self[-1]) * self[-2] ) + np.sqrt(6) * ( np.conj(self[1]) * self[0] + np.conj(self[0]) * self[-1] )
    
    def localMagMinus(self):
        return np.conj( self.localMagPlus() )


class SpinorBECGroundState2D():
    def __init__(self, grid, params,  psi ):
        """
        Parameters:
        -----------
        grid : 2D Grid object
        params : dict with 'c0', 'c2', 'c4', 'trap', 'dt', 'q'
        psi : Spinor Object
        """
        self.grid                             = grid
        self.params:dict                      = params
        self.dx:float                         = grid.grid_spacing_x
        self.dy:float                         = grid.grid_spacing_y
        self.dt:float                         = params['dit'] if 'dit' in params.keys() else params['dt']
        self.waveFunctions: list[Spinor]      = [ psi ] # This will be the actual results over time
        self.trialWavefunctions: list[Spinor] = [] # This is a helper list
        self.tolerance: float                 = 1e-8
        
        # Stabilization parameters (tune these!)
        self.alpha2:float       = 0
        self.alpha1:float       = 0
        self.alpha0:float       = 0
        self.alpha_minus1:float = 0
        self.alpha_minus2:float = 0

        self.intDebug = True 
        self.gradDebug = False


    def fullStep( self ):
        self.trialWavefunctions = [self.waveFunctions[-1]]
        self.nonlinearStep()
        while abs( self.trialWavefunctions[-2] - self.trialWavefunctions[-1] ) > self.tolerance:
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
                         ) for j in [2, 1, 0, -1, -2]] )
        
        if self.intDebug:
            spaceTimeVector = Spinor(*[ 1/self.dt * psiPrevious[j] for j in [2,1,0,-1,-2]])

        constantPlus2 = vector[2] + spaceTimeVector[2]
        constantPlus1 = vector[1] + spaceTimeVector[1]
        constantZero = vector[0] + spaceTimeVector[0]
        constantMinus1 = vector[-1] + spaceTimeVector[-1]
        constantMinus2 = vector[-2] + spaceTimeVector[-2]

        nx = self.grid.shape[0]
        ny = self.grid.shape[1]
        shapeNum = nx * ny
        nyDiag = [-1/(4*self.dy**2)]*nx*(ny-1)
        nxDiag = ([-1/(4*self.dx**2)]*(nx-1) + [0]) * ny

        matrixPlus2 = diags( [ [self.alpha2 + 1/(self.dt) +1/(2*self.dx**2) + 1/(2*self.dy**2)]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr' )
        
        if self.intDebug:
            matrixPlus2 = diags([ [self.alpha1 + 1/(self.dt) ]*shapeNum, ], 
                             [0], shape = (shapeNum,shapeNum), format='csr')
        
        newPlus2 = spsolve( matrixPlus2, constantPlus2.flatten() ).reshape(self.grid.shape)

        matrixPlus1 = diags( [ [self.alpha1 + 1/(self.dt) +1/(2*self.dx**2) + 1/(2*self.dy**2)]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr' )
        
        if self.intDebug:
            matrixPlus1 = diags([ [self.alpha1 + 1/(self.dt) ]*shapeNum, ], 
                             [0], shape = (shapeNum,shapeNum), format='csr')
        
        newPlus1 = spsolve( matrixPlus1, constantPlus1.flatten() ).reshape(self.grid.shape)

        matrixZero = diags( [ [self.alpha0 + 1/(self.dt) +1/(2*self.dx**2) + 1/(2*self.dy**2)]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr'  )
        if self.intDebug:
            matrixZero = diags([ [self.alpha1 + 1/(self.dt) ]*shapeNum, ], 
                             [0], shape = (shapeNum,shapeNum), format='csr')
        newZero = spsolve( matrixZero, constantZero.flatten() ).reshape(self.grid.shape)

        matrixMinus1 = diags( [ [self.alpha_minus1 + 1/(self.dt) +1/(2*self.dx**2) + 1/(2*self.dy**2)]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr'  )
        
        if self.intDebug:
            matrixMinus1 = diags([ [self.alpha1 + 1/(self.dt) ]*shapeNum, ], 
                             [0], shape = (shapeNum,shapeNum), format='csr')
        
        newMinus1 = spsolve( matrixMinus1, constantMinus1.flatten() ).reshape(self.grid.shape)

        matrixMinus2 = diags( [ [self.alpha_minus2 + 1/(self.dt) +1/(2*self.dx**2) + 1/(2*self.dy**2)]*shapeNum, nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr'  )
        
        if self.intDebug:
            matrixMinus2 = diags([ [self.alpha1 + 1/(self.dt) ]*shapeNum, ], 
                             [0], shape = (shapeNum,shapeNum), format='csr')
        
        newMinus2 = spsolve( matrixMinus2, constantMinus2.flatten() ).reshape(self.grid.shape)

        self.trialWavefunctions.append( Spinor( newPlus2, newPlus1, newZero, newMinus1, newMinus2 ) )


    def nonLinearCalc( self, mu, lam ):
        psiPrevious = self.waveFunctions[-1]
        psiCurrent = self.trialWavefunctions[-1]
        psiHalf = Spinor( *[ 0.5*(psiPrevious[j] + psiCurrent[j]) for j in [2, 1, 0, -1, -2]] )

        psiPlus2 = ( self.alpha2 * psiCurrent[2]
                    - self.params['c0']/2 * ( psiCurrent.localNumber() + psiPrevious.localNumber() ) * psiHalf[2]
                    - self.params['c2'] * (psiCurrent.localMag() + psiPrevious.localMag() ) * psiHalf[2]
                    - ( self.params['trap'] + 4 * self.params['q'] ) * psiHalf[2]
                    - self.params['c2']/2 * (psiCurrent.localMagMinus() + psiPrevious.localMagMinus()) * psiHalf[1]
                    - self.params['c4']/(2*np.sqrt(5))  * (psiCurrent.localSpinSinglet() + psiPrevious.localSpinSinglet())* np.conj(psiHalf[-2])
                    + ( mu + 2 * lam ) * psiHalf[2]
                )

        psiPlus1 = ( self.alpha1 * psiCurrent[1]
                    - self.params['c0']/2 * ( psiCurrent.localNumber() + psiPrevious.localNumber() ) * psiHalf[1]
                    - self.params['c2']/2 * (psiCurrent.localMag() + psiPrevious.localMag() ) * psiHalf[1]
                    - ( self.params['trap'] +  self.params['q'] ) * psiHalf[1]
                    - self.params['c2']/2 *( np.sqrt(6)/2 * ( psiCurrent.localMagMinus() + psiPrevious.localMagMinus() ) * psiHalf[0] 
                                            +( psiCurrent.localMagPlus() + psiPrevious.localMagPlus() ) * psiHalf[2] )
                    + self.params['c4']/(2*np.sqrt(5))  * (psiCurrent.localSpinSinglet() + psiPrevious.localSpinSinglet())* np.conj(psiHalf[-1])
                    + ( mu + lam ) * psiHalf[1]
                )
        psiZero = ( self.alpha0 * psiCurrent[0]
                    - self.params['c0']/2 * ( psiCurrent.localNumber() + psiPrevious.localNumber() ) * psiHalf[0]
                    - ( self.params['trap'] ) * psiHalf[0]
                    - self.params['c2']*np.sqrt(6)/4 *( ( psiCurrent.localMagMinus() + psiPrevious.localMagMinus() ) * psiHalf[-1] 
                                            + ( psiCurrent.localMagPlus() + psiPrevious.localMagPlus() ) * psiHalf[1] )
                    - self.params['c4']/(2*np.sqrt(5))  * (psiCurrent.localSpinSinglet() + psiPrevious.localSpinSinglet())* np.conj(psiHalf[0])
                    + mu  * psiHalf[0]
                )
        psiMinus1 = ( self.alpha_minus1 * psiCurrent[-1]
                    - self.params['c0']/2 * ( psiCurrent.localNumber() + psiPrevious.localNumber() ) * psiHalf[-1]
                    + self.params['c2']/2 * (psiCurrent.localMag() + psiPrevious.localMag() ) * psiHalf[-1]
                    - ( self.params['trap'] + self.params['q'] ) * psiHalf[-1]
                    - self.params['c2']/2 *( np.sqrt(6)/2 * ( psiCurrent.localMagPlus() + psiPrevious.localMagPlus() ) * psiHalf[0] 
                                            + ( psiCurrent.localMagMinus() + psiPrevious.localMagMinus() ) * psiHalf[-2] )
                    + self.params['c4']/(2*np.sqrt(5))  * (psiCurrent.localSpinSinglet() + psiPrevious.localSpinSinglet())* np.conj(psiHalf[1])
                    + ( mu - lam ) * psiHalf[-1]
                )
        psiMinus2 = ( self.alpha_minus2 * psiCurrent[-2]
                    - self.params['c0']/2 * ( psiCurrent.localNumber() + psiPrevious.localNumber() ) * psiHalf[-2]
                    + self.params['c2'] * (psiCurrent.localMag() + psiPrevious.localMag() ) * psiHalf[-2]
                    - ( self.params['trap'] + 4 * self.params['q'] ) * psiHalf[-2]
                    - self.params['c2']/2 * (psiCurrent.localMagPlus() + psiPrevious.localMagPlus()) * psiHalf[-1]
                    - self.params['c4']/(2*np.sqrt(5))  * (psiCurrent.localSpinSinglet() + psiPrevious.localSpinSinglet())* np.conj(psiHalf[2])
                    + ( mu - 2 * lam ) * psiHalf[-2]
                )
        
     
        if self.gradDebug:
            return Spinor( self.alpha2 * psiCurrent[2] + (mu + 2 * lam) * psiHalf[2],
                           self.alpha1 * psiCurrent[1] + (mu + lam)*psiHalf[1], 
                           self.alpha0 * psiCurrent[0] + mu * psiHalf[0], 
                           self.alpha_minus1 * psiCurrent[-1] + (mu - lam)*psiHalf[-1],
                           self.alpha_minus2 * psiCurrent[-2] + (mu - 2 * lam) * psiHalf[-2] )

        return Spinor( psiPlus2, psiPlus1, psiZero, psiMinus1, psiMinus2 )

    def computeChemicalPotentials(self):
        """
        Compute μ and λ, The chemical and magnetic potentials
        These ensure mass and magnetization conservation
        """
        psiPrevious = self.waveFunctions[-1]
        psiCurrent = self.trialWavefunctions[-1]

        psiHalf = Spinor( *[0.5*(psiPrevious[j] + psiCurrent[j]) for j in [2, 1, 0, -1, -2]] )
        
        
        nHalf = psiHalf.number()
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

        posIndependentTermD = ( 
                                self.params['c0']/2 * (psiCurrent.localNumber() + psiPrevious.localNumber() )* psiHalf.localNumber()
                               + self.params['c2']/2 * ( psiCurrent.localMag() + psiPrevious.localMag() ) * psiHalf.localMag() 
                               + self.params['q'] * ( 4 * abs(psiHalf[2])**2 + abs(psiHalf[1])**2 + abs(psiHalf[-1])**2 + 4 * abs(psiHalf[-2])**2 )
                               + (
                                   self.params['c2']/2 * ( (psiCurrent.localMagPlus() + psiPrevious.localMagPlus() ) * psiHalf.localMagMinus() )
                                + self.params['c4']/2 * (psiCurrent.localSpinSinglet() + psiPrevious.localSpinSinglet() ) * np.conj( psiHalf.localSpinSinglet() ) 
                                ).real )
        
        rolledSpinorX = Spinor( rollWithZeros(psiHalf[2],2,axis=0), rollWithZeros(psiHalf[1],1,axis=0), rollWithZeros(psiHalf[0],1,axis=0), 
                               rollWithZeros(psiHalf[-1],1,axis=0), rollWithZeros(psiHalf[-2],-2,axis=0) )
        
        rolledSpinorY = Spinor( rollWithZeros(psiHalf[2],2,axis=1), rollWithZeros(psiHalf[1],1,axis=1), rollWithZeros(psiHalf[0],1,axis=1), 
                               rollWithZeros(psiHalf[-1],1,axis=1), rollWithZeros(psiHalf[-2],-2,axis=1) )

        gradTermD = ( (1/(2*self.dx**2))*( abs(rolledSpinorX[2]-psiHalf[2])**2 + abs(rolledSpinorX[1]-psiHalf[1])**2 + abs(rolledSpinorX[0]-psiHalf[0])**2 
                                          + abs(rolledSpinorX[-1]-psiHalf[-1])**2 + abs(rolledSpinorX[-2]-psiHalf[-2])**2  )
                     + (1/(2*self.dy**2))*( abs(rolledSpinorY[2]-psiHalf[2])**2+ abs(rolledSpinorY[1]-psiHalf[1])**2 + abs(rolledSpinorY[0]-psiHalf[0])**2 
                                           + abs(rolledSpinorY[-1]-psiHalf[-1])**2 + abs(rolledSpinorY[-2]-psiHalf[-2])**2) )

        trapTermD = self.params['trap'] * psiHalf.localNumber()

        posIndependentTermF = ( 
                                self.params['c0']/2 * ( psiCurrent.localNumber() + psiPrevious.localNumber() ) * psiHalf.localMag()
                               + self.params['c2']/2 * ( psiCurrent.localMag() + psiPrevious.localMag() )* psiHalf.localZeeman()
                               + self.params['q'] * ( 8 * abs(psiHalf[2])**2 + abs(psiHalf[1])**2 - abs(psiHalf[-1])**2 - 8 * abs(psiHalf[-2])**2 )
                                 + self.params['c2']/2 * ( 
                                   (psiCurrent.localMagMinus() + psiPrevious.localMagMinus() ) 
                                 * (2*np.conj(psiHalf[2])*psiHalf[1] + np.sqrt(6)/2 * np.conj(psiHalf[1])*psiHalf[0] - np.conj(psiHalf[-1])*psiHalf[-2] )
                                 + (psiCurrent.localMagPlus() + psiPrevious.localMagPlus()) 
                                 * (np.conj(psiHalf[1])*psiHalf[2] - np.sqrt(6)/2 * np.conj(psiHalf[-1])*psiHalf[0] - 2 * np.conj(psiHalf[-2])*psiHalf[-1] ) ).real  
                                 )

        gradTermF =  ( (1/(2*self.dx**2))*( 2 * abs(rolledSpinorX[2]-psiHalf[2])**2 + abs(rolledSpinorX[1]-psiHalf[1])**2 
                                           - (abs(rolledSpinorX[-1]-psiHalf[-1])**2 + 2 * abs(rolledSpinorX[-2]-psiHalf[-2])**2)  ) + 
                      (1/(2*self.dy**2))*( 2* abs(rolledSpinorY[2]-psiHalf[2])**2 + abs(rolledSpinorY[1]-psiHalf[1])**2 
                                          - ( abs(rolledSpinorY[-1]-psiHalf[-1])**2 + 2 * abs(rolledSpinorY[-2]-psiHalf[-2])**2 ) ) ) 

        trapTermF = self.params['trap'] * psiHalf.localMag() 

        if self.gradDebug:
            return ( np.sum( gradTermD ), np.sum( gradTermF ) )
        
        if self.intDebug:
            return ( np.sum( posIndependentTermD + trapTermD ), np.sum( posIndependentTermF + trapTermF ) )

        return ( np.sum( gradTermD + trapTermD + posIndependentTermD ), np.sum( gradTermF + trapTermF + posIndependentTermF ) )

