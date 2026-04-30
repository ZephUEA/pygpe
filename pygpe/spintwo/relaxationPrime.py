import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

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
        self.waveFunction:Spinor              = psi 
        


    def fullStep( self ):
        tempWfn = self.nonlinearStep()
        self.waveFunction = self.projection(tempWfn)
    
    def nonlinearStep( self ):
        psi = self.waveFunction

        selfInteractionTerm = ( self.params['trap'] + self.params['c0']*psi.localNumber() ) 

        numericalStabTerm = Spinor(
            self.params['c2']*(2*abs(psi[1])**2 + 2 * psi.localMag()) + 2/5 * self.params['c4']*(abs(psi[-2])**2) 
            + 4 * self.params['q'],

            self.params['c2'] * ( 2* abs(psi[2])**2 + psi.localMag() + 3 * abs(psi[0])**2 ) 
                + 2/5 * self.params['c4']*abs(psi[-1]) + self.params['q'],

            3*self.params['c2']*(abs(psi[1])**2 + abs(psi[-1])**2) + self.params['c4']/5 * abs(psi[0])**2,

            self.params['c2'] * ( 2* abs(psi[-2])**2 - psi.localMag() + 3 * abs(psi[0])**2 ) 
                + 2/5 * self.params['c4']*abs(psi[1]) + self.params['q'],

            self.params['c2']*(2*abs(psi[-1])**2 - 2 * psi.localMag()) + 2/5 * self.params['c4']*(abs(psi[2])**2)
            + 4 * self.params['q']
        )

        interactionTerm = Spinor(
            self.params['c2'] * ( psi.localMagMinus() - 2 * np.conj(psi[1])*psi[2]  ) * psi[1]
            + self.params['c4']/5 * (psi[0]**2 - 2 * psi[1]*psi[-1])*np.conj(psi[-2]),

            self.params['c2'] * ( (psi.localMagPlus() - 2 * np.conj(psi[2])* psi[1] ) * psi[2] 
                                 + (np.sqrt(6)/2 * psi.localMagMinus() - 3 * np.conj(psi[0])*psi[1]) * psi[0])
            - self.params['c4']/5 * (2*psi[2]*psi[-2]+ psi[0]**2)*np.conj(psi[-1]),

            self.params['c2'] * ( (np.sqrt(6)/2 * psi.localMagPlus() - 3 * np.conj(psi[1]) * psi[0] )*psi[1] 
                                 + (np.sqrt(6)/2 * psi.localMagMinus() - 3 * np.conj(psi[-1]) * psi[0] )*psi[-1])

            + self.params['c4'] * 2/5 * (psi[2]*psi[-2] - psi[1]*psi[-1]) * np.conj(psi[0]),

            self.params['c2'] * ( (psi.localMagMinus() - 2 * np.conj(psi[-2])* psi[-1] ) * psi[-2] 
                                 + (np.sqrt(6)/2 * psi.localMagPlus() - 3 * np.conj(psi[0])*psi[-1]) * psi[0] )
            - self.params['c4']/5 * (2*psi[2]*psi[-2]+ psi[0]**2)*np.conj(psi[1]),

            self.params['c2']* ( psi.localMagPlus() - 2 * np.conj(psi[-1])*psi[-2]  ) * psi[-1]
            + self.params['c4']/5 * (psi[0]**2 - 2 * psi[1]*psi[-1])*np.conj(psi[2])
        )

        # Set up linear algebra problem Ax = b

        constant = Spinor(*[psi[i]/self.dt - interactionTerm[i] for i in [2,1,0,-1,-2]])

        nx = self.grid.shape[0]
        ny = self.grid.shape[1]
        shapeNum = nx * ny
        nyDiag = [-1/(2*self.dy**2)]*nx*(ny-1)
        nxDiag = ([-1/(2*self.dx**2)]*(nx-1) + [0]) * ny

        matrixPlus2 = diags( [ [ 1/self.dt + 1/(self.dx**2) + 1/(self.dy**2) ]*shapeNum + selfInteractionTerm.flatten() + numericalStabTerm[2].flatten(), nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr' )

        tmpP2 = spsolve( matrixPlus2, constant[2].flatten() ).reshape(self.grid.shape)

        matrixPlus1 = diags( [ [ 1/self.dt + 1/(self.dx**2) + 1/(self.dy**2) ]*shapeNum + selfInteractionTerm.flatten() + numericalStabTerm[1].flatten(), nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr' )

        tmpP1 = spsolve( matrixPlus1, constant[1].flatten() ).reshape(self.grid.shape)

        matrixZero = diags([ [ 1/self.dt +  1/(self.dx**2) + 1/(self.dy**2) ]*shapeNum + selfInteractionTerm.flatten() + numericalStabTerm[0].flatten() , nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr' )

        tmp0 = spsolve( matrixZero, constant[0].flatten() ).reshape(self.grid.shape)

        matrixMinus1 = diags( [ [ 1/self.dt + 1/(self.dx**2) + 1/(self.dy**2) ]*shapeNum + selfInteractionTerm.flatten() + numericalStabTerm[-1].flatten(), nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr' )

        tmpM1 = spsolve( matrixMinus1, constant[-1].flatten() ).reshape(self.grid.shape)

        matrixMinus2 = diags( [ [ 1/self.dt + 1/(self.dx**2) + 1/(self.dy**2) ]*shapeNum + selfInteractionTerm.flatten() + numericalStabTerm[-2].flatten(), nxDiag, nxDiag, nyDiag, nyDiag], 
                             [0,1,-1,nx, -nx], shape = (shapeNum,shapeNum), format='csr' )

        tmpM2 = spsolve( matrixMinus2, constant[-2].flatten() ).reshape(self.grid.shape)

        return Spinor( tmpP2, tmpP1, tmp0, tmpM1, tmpM2 )




    def projection( self, psi ):
        mag = self.waveFunction.mag()
        num = self.waveFunction.number()
        magNorm = mag / num

        if magNorm == 1 and not np.any([np.sum(abs(psi[i])**2) for i in [2,0,-1,-2]]):
            oneProjector = np.sqrt(num/np.sum(abs(psi[1])**2))
            projectedSpinor = Spinor(psi[2],oneProjector*psi[1],psi[0],psi[-1],psi[-2] )
            return projectedSpinor
        
        if magNorm == 0 and not np.any([np.sum(abs(psi[i])**2) for i in [2,1,-1,-2]]):
            zeroProjector = np.sqrt(num/np.sum(abs(psi[0])**2))
            projectedSpinor = Spinor(psi[2],psi[1],zeroProjector*psi[0],psi[-1],psi[-2] )
            return projectedSpinor
        
        polynomial = np.array([
            (2-magNorm)*np.sum(abs(psi[2])**2 ),
            (1-magNorm)*np.sum(abs(psi[1])**2 ),
            (-magNorm)*np.sum(abs(psi[0])**2 ),
            (-1-magNorm)*np.sum(abs(psi[-1])**2 ),
            (-2-magNorm)*np.sum(abs(psi[-2])**2 )
            ])
        
        roots = np.roots(polynomial)
        root = max( roots )
        
        # OUR ISSUE IS HERE
        zeroProjector = np.sqrt(1/(sum([root**(i) * np.sum(abs(psi[i])**2)/num for i in [2,1,0,-1,-2]])))

        projectedSpinor = Spinor(
            psi[2] * zeroProjector * root,
            psi[1] * zeroProjector * np.sqrt( root ),
            psi[0] * zeroProjector,
            psi[-1] * zeroProjector / np.sqrt( root ),
            psi[-2] * zeroProjector / root
        )
        # print(f'N={num}\nM={mag}')
        # print( f'N={projectedSpinor.number()}\nM={projectedSpinor.mag()}' )
        return projectedSpinor

  