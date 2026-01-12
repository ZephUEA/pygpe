try:
    import cupy as cp # type: ignore
except ImportError:
    import numpy as cp

import os

from pygpe.shared.grid import Grid
from pygpe.shared.wavefunction import _Wavefunction
import matplotlib.pyplot as plt
import scipy.linalg as scl
import animation as ani


class SpinOneWavefunction4D(_Wavefunction):
    def __init__(self, grid:Grid, real:bool=False):
        super().__init__(grid)
        
        self.up_A_component = cp.zeros(grid.shape, dtype="complex128")
        self.up_B_component = cp.zeros(grid.shape, dtype="complex128")
        self.down_A_component = cp.zeros(grid.shape, dtype="complex128")
        self.down_B_component = cp.zeros(grid.shape, dtype="complex128")
        self.fourier_up_A_component = cp.zeros(grid.shape, dtype="complex128")
        self.fourier_up_B_component = cp.zeros(grid.shape, dtype="complex128")
        self.fourier_down_A_component = cp.zeros(grid.shape, dtype="complex128")
        self.fourier_down_B_component = cp.zeros(grid.shape, dtype="complex128")

        if real:
            self.diagonalise()

        self.atom_num_up_A = 0
        self.atom_num_up_B = 0
        self.atom_num_down_A = 0
        self.atom_num_down_B = 0
    
    def diagonalise(self):
        spin_A_plus,spin_A_minus,spin_F_A,spin_B_plus,spin_B_minus,spin_F_B = self.calculate_spins()
        shape = self.grid.shape
    
        matrix = cp.moveaxis(cp.array([[spin_F_A, spin_B_minus, spin_A_minus, cp.zeros(shape)],[spin_B_plus, spin_F_B, cp.zeros(shape), spin_A_minus],
                       [spin_A_plus, cp.zeros(shape), -spin_F_B, spin_B_minus],[cp.zeros(shape), spin_A_plus, spin_B_plus, -spin_F_A]]), [0,1], [-2,-1] )

        self.eigenvalues, self.eigenvectors = cp.linalg.eigh( matrix )
        self.eigenvectorsInverse = cp.linalg.inv( self.eigenvectors )

    
    def fft(self) -> None:
        """Fourier transforms real-space components and updates Fourier-space
        components.
        """
        self.fourier_up_A_component = cp.fft.fftn(self.up_A_component)
        self.fourier_up_B_component = cp.fft.fftn(self.up_B_component)
        self.fourier_down_A_component = cp.fft.fftn(self.down_A_component)
        self.fourier_down_B_component = cp.fft.fftn(self.down_B_component)

    def ifft(self) -> None:
        """Inverse Fourier transforms Fourier-space components and updates
        real-space components.
        """
        self.up_A_component = cp.fft.ifftn(self.fourier_up_A_component)
        self.up_B_component = cp.fft.ifftn(self.fourier_up_B_component)
        self.down_A_component = cp.fft.ifftn(self.fourier_down_A_component)
        self.down_B_component = cp.fft.ifftn(self.fourier_down_B_component)
    
    def add_noise( self, mean: float, std_dev: float ) -> None:
        self.up_A_component += super()._generate_complex_normal_dist( mean, std_dev )
        self.up_B_component += super()._generate_complex_normal_dist( mean, std_dev )
        self.down_A_component += super()._generate_complex_normal_dist( mean, std_dev )
        self.down_B_component += super()._generate_complex_normal_dist( mean, std_dev )

        self._update_atom_numbers()

    def set_wavefunction(self, up_A_component:cp.ndarray=None, up_B_component:cp.ndarray=None, down_A_component:cp.ndarray=None, down_B_component:cp.ndarray=None):
        if up_A_component is not None:
            self.up_A_component = up_A_component
        if up_B_component is not None:
            self.up_B_component = up_B_component
        if down_A_component is not None:
            self.down_A_component = down_A_component
        if down_B_component is not None:
            self.down_B_component = down_B_component 

        self._update_atom_numbers()

    def apply_phase(
        self, phase: cp.ndarray, components: str | list[str] = "all"
    ) -> None:
        """Applies a phase to specified components.

        :param phase: The phase to be applied.
        :param components: "all", "plus", "zero", "minus" or a list of strings
            specifying the required components.
        """
        match components:
            case [*_]:
                for component in components:
                    self._apply_phase_to_component(phase, component)
            case "all":
                for component in ["up_a", "up_b", "down_b", "down_a"]:
                    self._apply_phase_to_component(phase, component)
            case str(component):
                self._apply_phase_to_component(phase, component)
            case _:
                raise ValueError(f"Components type {components} is unsupported")

    def _apply_phase_to_component(self, phase: cp.ndarray, component: str) -> None:
        """Applies the specified phase to the specified component."""
        match component.lower():
            case "up_a":
                self.up_A_component *= cp.exp(1j * phase)
            case "up_b":
                self.up_B_component *= cp.exp(1j * phase)
            case "down_a":
                self.down_A_component *= cp.exp(1j * phase)
            case "down_b":
                self.down_B_component *= cp.exp(1j * phase)
            case _:
                raise ValueError(f"Component type {component} is unsupported")
    
    def density(self) -> cp.ndarray:
        """Returns an array of the total condensate density.

        :return: Total condensate density.
        """
        return (
            cp.abs(self.up_A_component) ** 2
            + cp.abs(self.up_B_component) ** 2
            + cp.abs(self.down_A_component) ** 2
            + cp.abs(self.down_B_component) ** 2
        )
    
    def _update_atom_numbers(self) -> None:
        self.atom_num_up_A = self.grid.grid_spacing_product * cp.sum(
            cp.abs(self.up_A_component) ** 2
        )
        self.atom_num_up_B = self.grid.grid_spacing_product * cp.sum(
            cp.abs(self.up_B_component) ** 2
        )
        self.atom_num_down_A = self.grid.grid_spacing_product * cp.sum(
            cp.abs(self.down_A_component) ** 2
        )
        self.atom_num_down_B = self.grid.grid_spacing_product * cp.sum(
            cp.abs(self.down_B_component) ** 2
        )

    def calculate_spins(self ) -> tuple[cp.ndarray, cp.ndarray]:
        """Calculates the A and B system spins.

        :param wfn: The wavefunction of the system.
        :return: A system, then B system spins (A_+,A_-,A_3,B_+,B_-,B_3).
        """
        spin_A_plus = cp.conj( self.up_A_component ) * self.down_B_component + cp.conj( self.up_B_component ) * self.down_A_component
        spin_A_minus = cp.conj( spin_A_plus )
        spin_F_A = cp.abs(self.up_A_component)**2 - cp.abs(self.down_A_component)**2 
        spin_B_plus = cp.conj(self.up_A_component) * self.up_B_component + cp.conj(self.down_B_component) * self.down_A_component
        spin_B_minus = cp.conj(spin_B_plus)
        spin_F_B = cp.abs(self.up_B_component)**2 - cp.abs(self.down_B_component)**2 

        return (spin_A_plus,spin_A_minus,spin_F_A,spin_B_plus,spin_B_minus,spin_F_B)



def step_wavefunction(wfn: SpinOneWavefunction4D, params: dict) -> None:
    """Propagates the wavefunction forward one time step.

    :param wfn: The wavefunction of the system.
    :type wfn: :class:`Wavefunction`
    :param params: The parameters of the system.
    :type params: dict
    """
    _kinetic_zeeman_step(wfn, params)
    wfn.ifft()
    _interaction_step(wfn, params)
    wfn.fft()
    _kinetic_zeeman_step(wfn, params)
    if isinstance(params["dt"], complex):
        _renormalise_wavefunction(wfn)

def _kinetic_zeeman_step(wfn:SpinOneWavefunction4D, params:dict) -> None:
    wfn.up_A_component *= cp.exp(-0.25 * 1j * params["dt"] * wfn.grid.wave_number)
    wfn.up_B_component *= cp.exp(-0.25 * 1j * params["dt"] * wfn.grid.wave_number)
    wfn.down_A_component *= cp.exp(-0.25 * 1j * params["dt"] * wfn.grid.wave_number)
    wfn.down_B_component *= cp.exp(-0.25 * 1j * params["dt"] * wfn.grid.wave_number)



def _interaction_step( wfn:SpinOneWavefunction4D, params:dict ) -> None:
    
    scalar = params['dt'] * -1j * params["c2"]
    density = _calculate_density( wfn )
    shape = wfn.grid.shape  
    if isinstance( params['dt'], complex ):
        spin_A_plus,spin_A_minus,spin_F_A,spin_B_plus,spin_B_minus,spin_F_B = wfn.calculate_spins()
    
        matrix = cp.moveaxis(cp.array([[spin_F_A, spin_B_minus, spin_A_minus, cp.zeros(shape)],[spin_B_plus, spin_F_B, cp.zeros(shape), spin_A_minus],
                       [spin_A_plus, cp.zeros(shape), -spin_F_B, spin_B_minus],[cp.zeros(shape), spin_A_plus, spin_B_plus, -spin_F_A]]), [0,1], [-2,-1] )
        exponentiatedMatrix = cp.moveaxis( scl.expm( scalar * matrix ), [-2,-1], [0,1] )
    
    else:
        evolvedDiagonal = cp.exp( scalar * wfn.eigenvalues )
        exponentiatedDiagonal = cp.zeros( (*shape, 4, 4 ), dtype='complex128' )
        for i in range(4):
            exponentiatedDiagonal[:,:,i,i] = evolvedDiagonal[:,:,i]
        exponentiatedMatrix = cp.moveaxis( cp.matmul( wfn.eigenvectors, cp.matmul( exponentiatedDiagonal, wfn.eigenvectorsInverse ) ),[-2,-1], [0,1] )
    

    temp_wfn_up_A = ( exponentiatedMatrix[0,0] * wfn.up_A_component + exponentiatedMatrix[0,1] * wfn.up_B_component 
                    + exponentiatedMatrix[0,2] * wfn.down_B_component + exponentiatedMatrix[0,3] * wfn.down_A_component )
    temp_wfn_up_B = ( exponentiatedMatrix[1,0] * wfn.up_A_component + exponentiatedMatrix[1,1] * wfn.up_B_component 
                    + exponentiatedMatrix[1,2] * wfn.down_B_component + exponentiatedMatrix[1,3] * wfn.down_A_component )
    temp_wfn_down_B = ( exponentiatedMatrix[2,0] * wfn.up_A_component + exponentiatedMatrix[2,1] * wfn.up_B_component 
                    + exponentiatedMatrix[2,2] * wfn.down_B_component + exponentiatedMatrix[2,3] * wfn.down_A_component )
    temp_wfn_down_A =  ( exponentiatedMatrix[3,0] * wfn.up_A_component + exponentiatedMatrix[3,1] * wfn.up_B_component 
                    + exponentiatedMatrix[3,2] * wfn.down_B_component + exponentiatedMatrix[3,3] * wfn.down_A_component )
    
    wfn.up_A_component = temp_wfn_up_A * cp.exp(-1j * params["c0"] * density * params['dt'] )
    wfn.up_B_component = temp_wfn_up_B * cp.exp(-1j * params["c0"] * density * params['dt'] )
    wfn.down_B_component = temp_wfn_down_B * cp.exp(-1j * params["c0"] * density * params['dt'] )
    wfn.down_A_component = temp_wfn_down_A * cp.exp(-1j * params["c0"] * density * params['dt'] )


def _calculate_density(wfn: SpinOneWavefunction4D) -> cp.ndarray:
    """Calculates the total condensate density.

    :param wfn: The wavefunction of the system.
    :return: The total atomic density.
    """
    return (
        cp.abs(wfn.up_A_component) ** 2
        + cp.abs(wfn.up_B_component) ** 2
        + cp.abs(wfn.down_A_component) ** 2
        + cp.abs(wfn.down_B_component) ** 2
    )


def _renormalise_wavefunction(wfn: SpinOneWavefunction4D) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    wfn.ifft()
    correct_atom_num = wfn.atom_num_up_A + wfn.atom_num_up_B + wfn.atom_num_down_A + wfn.atom_num_down_B
    current_atom_num = _calculate_atom_num(wfn)
    wfn.up_A_component *= cp.sqrt(correct_atom_num / current_atom_num)
    wfn.up_B_component *= cp.sqrt(correct_atom_num / current_atom_num)
    wfn.down_A_component *= cp.sqrt(correct_atom_num / current_atom_num)
    wfn.down_B_component *= cp.sqrt(correct_atom_num / current_atom_num)
    wfn.fft()


def _calculate_atom_num(wfn: SpinOneWavefunction4D) -> float:
    """Calculates the total atom number of the system.

    :param wfn: The wavefunction of the system.
    :return: The total atom number.
    """
    atom_num_up_A = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.up_A_component) ** 2
    )
    atom_num_up_B = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.up_B_component) ** 2
    )
    atom_num_down_A = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.down_A_component) ** 2
    )
    atom_num_down_B = wfn.grid.grid_spacing_product * cp.sum(
        cp.abs(wfn.down_B_component) ** 2
    )

    return atom_num_up_A + atom_num_down_A + atom_num_up_B + atom_num_down_B


def createWavefunction( grid:Grid, params:dict ) -> tuple[cp.ndarray, ...]:
    shapex = grid.shape[0]
    shapey = grid.shape[1]
    aup =  cp.concat( (cp.zeros(( shapex, shapey // 2), dtype='complex128'), cp.ones( ( shapex, shapey // 2), dtype='complex128' ) * params['n0'] / cp.sqrt( 2 )), axis=1 )
    adown = cp.concat( (cp.ones(( shapex, shapey // 2), dtype='complex128') * params['n0'] / cp.sqrt( 2 ), cp.zeros( ( shapex, shapey // 2), dtype='complex128' )), axis=1 )
    bup = cp.zeros( grid.shape, dtype='complex128' ) * params['n0'] / 2
    bdown = cp.zeros( grid.shape, dtype='complex128' ) * params['n0'] / 2
    return aup, adown, bup, bdown


def phaseBoundary( grid:Grid, params:dict ) -> SpinOneWavefunction4D:
    psi = SpinOneWavefunction4D( grid )
    shape = grid.shape
    phaseWidth = shape[0] 
    phaseHeight = shape[1] // 2
    upPhase = cp.concatenate( ( cp.zeros( ( phaseWidth, phaseHeight ), dtype="complex128" ), cp.ones( ( phaseWidth, phaseHeight ), dtype="complex128" ) ), axis=1 ) * cp.sqrt( params[ "n0" ] )
    downPhase = cp.concatenate( ( cp.ones( ( phaseWidth, phaseHeight ), dtype="complex128" ), cp.zeros( ( phaseWidth, phaseHeight ), dtype="complex128" ) ), axis = 1 ) * cp.sqrt( params[ "n0" ] )
    psi.set_wavefunction( up_A_component=upPhase, down_B_component=downPhase )
    return psi 

def kelvinHelmholtzBoundary( grid:Grid, params:dict, cycles:int ) -> SpinOneWavefunction4D:
    psi = phaseBoundary( grid, params )
    phase = cp.linspace( 0, 2 * cycles * cp.pi, grid.num_points_x )
    phaseReversed = phase[::-1]
    psi.apply_phase( phase[:,None], 'up_a' )
    psi.apply_phase( phaseReversed[:,None], 'down_b' )
    return psi



def getData( wfn, params ):
    aupMag = []
    adownMag = []
    bupMag = []
    bdownMag =[]
    gridProd = cp.prod( wfn.grid.shape )
    ts = cp.linspace( 0, params['nt'] * abs( params['dt'] ), params['nt'] // params['frameRate'] + 1 )[:-1]
    for step in range( params['nt'] ):
        if step % params['frameRate'] == 0:
            print( params['t'] )
            aupMag.append(   ( wfn.up_A_component ) / 1 )
            adownMag.append(  ( wfn.down_A_component ) / 1 )
            bupMag.append(  ( wfn.up_B_component  ) / 1 )
            bdownMag.append(  ( wfn.down_B_component  ) / 1 )
            
        step_wavefunction( wfn, params )
        params["t"] += params["dt"]
         
    return {'times':ts,  'aUp':aupMag,  'aDown':adownMag,  'bUp':bupMag,  'bDown':bdownMag}


def takeFrame(frames_dir, params, frame, results):
    fig,axs = plt.subplots(2,4, figsize=(36,24))
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"

    up_A = axs[0][0].pcolormesh(
            ( xMesh ),
            ( yMesh ),
            ( abs( results['aUp'][frame] ) ** 2 ), 
            vmin=0, vmax=1 )
    fig.colorbar( up_A )
    axs[0][0].set_title(r'$|A_\uparrow|^2$')
    down_A = axs[0][2].pcolormesh(
            ( xMesh ),
            ( yMesh ),
            ( abs( results['aDown'][frame] ) ** 2 ), 
            vmin=0, vmax=1 )
    fig.colorbar( down_A )
    axs[0][2].set_title(r'$|A_\downarrow|^2$')
    up_B = axs[1][0].pcolormesh(
            ( xMesh ),
            ( yMesh ),
            ( abs( results['bUp'][frame] ) ** 2 ), 
            vmin=0, vmax=1 )
    fig.colorbar( up_B )
    axs[1][0].set_title(r'$|B_\uparrow|^2$')
    down_B = axs[1][2].pcolormesh(
            ( xMesh ),
            ( yMesh ),
            ( abs( results['bDown'][frame] ) ** 2 ), 
            vmin=0, vmax=1 )
    fig.colorbar( down_B )
    axs[1][2].set_title(r'$|B_\downarrow|^2$')
    up_A_ang = axs[0][1].pcolormesh(
            ( xMesh ),
            ( yMesh ),
            ( cp.angle( results['aUp'][frame] ) ), 
            vmin=-cp.pi, vmax=cp.pi )
    fig.colorbar( up_A_ang )
    axs[0][1].set_title(r'$\arg(A_\uparrow)$')
    down_A_ang = axs[0][3].pcolormesh(
            ( xMesh ),
            ( yMesh ),
            ( cp.angle( results['aDown'][frame] ) ), 
            vmin=-cp.pi, vmax=cp.pi )
    fig.colorbar( down_A_ang )
    axs[0][3].set_title(r'$\arg(A_\downarrow)$')
    up_B_ang = axs[1][1].pcolormesh(
            ( xMesh ),
            ( yMesh ),
            ( cp.angle( results['bUp'][frame] ) ), 
            vmin=-cp.pi, vmax=cp.pi )
    fig.colorbar( up_B_ang )
    axs[1][1].set_title(r'$\arg(B_\uparrow)$')
    down_B_ang = axs[1][3].pcolormesh(
            ( xMesh ),
            ( yMesh ),
            ( cp.angle( results['bDown'][frame] ) ), 
            vmin=-cp.pi, vmax=cp.pi )
    fig.colorbar( down_B_ang )
    axs[1][3].set_title(r'$\arg(B_\downarrow)$')
    
    fig.suptitle(f'Components of ground states: {'Ferromagnetic' if params['c2'] < 0 else 'Polar'}')
    plt.savefig(frame_path)
    plt.close()

if __name__ == '__main__':
    gridPoints = (128,128)
    gridSpacings = (0.5,0.5)
    grid = Grid(gridPoints, gridSpacings)
    params = {
        'c0' : 10,
        'c2' : -0.5,
        'n0' : 1, 
        'dt' : (1) * 1e-2, 
        'nt' : 300_000,
        't' : 0,
        'frameRate' : 1000 }
    
    # psi = SpinOneWavefunction4D(grid, not isinstance( params['dt'], complex ) )
    # components = createWavefunction( grid, params )
    # psi.set_wavefunction( up_A_component=components[0], down_A_component=components[1], up_B_component=components[2], down_B_component=components[3] )
    
    
    psi = kelvinHelmholtzBoundary( grid, params, 10 )
    
    
    psi.add_noise(0.0, 1e-4)

    psi.fft()
    psi.diagonalise()

    results = getData( psi, params )

    xs = cp.arange( -gridPoints[0]//2, gridPoints[0]//2 ) * gridSpacings[0] 
    ys = cp.arange( -gridPoints[1]//2, gridPoints[1]//2 ) * gridSpacings[1] 

    xMesh, yMesh = cp.meshgrid( xs, ys, indexing='ij' )
    

    frames_dir = 'frames'

    os.makedirs(frames_dir, exist_ok=True)

    for frame in range( len( results['times'] ) ):
        takeFrame( frames_dir, params, frame, results )


    # plt.legend([r'$\Re(A_\uparrow)$',r'$\Im(A_\uparrow)$', r'$\Re(A_\downarrow)$',r'$\Im(A_\downarrow)$', r'$\Re(B_\uparrow)$',r'$\Im(B_\uparrow)$', r'$\Re(B_\downarrow)$',r'$\Im(B_\downarrow)$'])
    # plt.xlabel('Imaginary Time')
    # plt.ylabel('Component')
    # plt.show()

    ani.movieFromFrames('GroundState.mp4', frames_dir  )