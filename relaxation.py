import time
import os
import h5py

try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
import pygpe.shared.vortices as vort

import pygpe.spinone as gpe

from correlation import pseudoVorticity
import animation as ani

######################################################Data helper functions###################################################################

def hdf5ReadScalars( hdf5Obj, prepend:str='', makeUnique:bool=False ) -> dict:
    resultsDict = {}
    for name, data in hdf5Obj.items():
        if isinstance( data, h5py.Group ):
            resultsDict.update( hdf5ReadScalars( data, name, makeUnique ) )
        else:
            if data.shape == ():
                dictName = name
                if makeUnique and prepend != '':
                    dictName = prepend + '_' + name
                resultsDict.update( { dictName : data[()] } )
    return resultsDict

######################################################Initial conditions###################################################################

def phaseBoundary( grid:gpe.Grid, params:dict ) -> gpe.SpinOneWavefunction:
    psi = gpe.SpinOneWavefunction( grid )
    shape = grid.shape
    phaseWidth = shape[0] 
    phaseHeight = shape[1] // 2
    upPhase = cp.concatenate( ( cp.zeros( ( phaseWidth, phaseHeight ), dtype="complex128" ), cp.ones( ( phaseWidth, phaseHeight ), dtype="complex128" ) ), axis=1 ) * cp.sqrt( params[ "n0" ] )
    downPhase = cp.concatenate( ( cp.ones( ( phaseWidth, phaseHeight ), dtype="complex128" ), cp.zeros( ( phaseWidth, phaseHeight ), dtype="complex128" ) ), axis = 1 ) * cp.sqrt( params[ "n0" ] )
    psi.set_wavefunction( plus_component=upPhase, minus_component=downPhase )
    return psi 

def kelvinHelmholtzBoundary( grid:gpe.Grid, params:dict, cycles:int ) -> gpe.SpinOneWavefunction:
    psi = phaseBoundary( grid, params )
    phase = cp.linspace( 0, 2 * cycles * cp.pi, grid.num_points_x )
    phaseReversed = phase[::-1]
    psi.apply_phase( phase[:,None], 'plus' )
    psi.apply_phase( phaseReversed[:,None], 'minus' )
    return psi

def phaseGradientCorrection( phase:cp.ndarray, axis:int=0 ) -> cp.ndarray:
    '''Force a phase profile to be periodic in the given axis'''
    newPhase = phase % ( 2 * cp.pi )
    first = cp.moveaxis( newPhase, axis, 0)[0,:]
    last = cp.moveaxis( newPhase, axis, 0 )[-1,:]
    delta = ( first - last )
    delta[ delta == 0 ] = cp.pi * 2
    return ( newPhase * 2 * cp.pi / cp.expand_dims( delta, axis=axis ) ) % ( 2 * cp.pi )

def vortexBoundaryPair( grid:gpe.Grid, params:dict, xSteps:int, ypos:int ) -> gpe.SpinOneWavefunction:
    '''Build two vorticies in the same phase separated along the axis parralel to the boundary'''
    psi = phaseBoundary( grid, params )
    phase = vort._calculate_vortex_contribution( grid, -xSteps, ypos, 1 ) + vort._calculate_vortex_contribution( grid, xSteps, ypos, 1 )
    psi.apply_phase( phase, [ "plus" ] )
    return psi

def initialEvolution( psi:gpe.SpinOneWavefunction, params:dict, nt:int, dt:float ) -> None:
    percent = nt / 100
    t0 = params["t"]
    for _ in range( nt ):
        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)
        params["t"] += dt  # Increment time count
        if nt % percent:
            print( f"{nt // percent}% through initial evolution" )

    params["t"] = t0 # Reset the time 
    return

def hqv( grid:gpe.Grid, params:dict, xpos:int, ypos:int ) -> gpe.SpinOneWavefunction:
    psi = gpe.SpinOneWavefunction( grid )
    ones = cp.sqrt( params[ "n0" ] / 2 ) * cp.ones( grid.shape, dtype="complex128" )
    psi.set_wavefunction( plus_component=ones, minus_component=ones )
    phase = vort._calculate_vortex_contribution( grid, xpos, ypos, 1 )
    psi.apply_phase( phase, [ "plus" ] )
    return psi

def polarVortexPair( grid:gpe.Grid, params:dict, xSteps:int ) -> gpe.SpinOneWavefunction:
    psi = gpe.SpinOneWavefunction( grid )
    middle = cp.sqrt( params["n0"] ) * cp.ones( grid.shape, dtype="complex128" )
    psi.set_wavefunction( zero_component=middle )
    phase = vort._calculate_vortex_contribution( grid, -xSteps, 0, 1 ) + vort._calculate_vortex_contribution( grid, xSteps, 0, -1 )
    psi.apply_phase( phase, ['zero'] )
    return psi

def majorityUp( grid, params, multiplicity ):
    psi = gpe.SpinOneWavefunction( grid )
    up = cp.ones( grid.shape, dtype="complex128" ) * cp.sqrt( params["n0"] * multiplicity / ( multiplicity + 1 ) )
    down = cp.ones( grid.shape, dtype="complex128" ) * cp.sqrt( params["n0"] / ( multiplicity + 1 ) )
    psi.set_wavefunction( plus_component=up, minus_component=down )
    return psi

def majorityUpVortex( grid, params, multiplicity ):
    psi = majorityUp( grid, params, multiplicity )
    phase = vort._calculate_vortex_contribution( grid, 0, 0, 1 )
    psi.apply_phase( phase, ['plus'] )
    return psi
    

######################################################Data generation###################################################################

def getData( psi, params:dict, fileName:str, dataPath:str ) -> None:
    data = gpe.DataManager(fileName, dataPath, psi, params)
    for i in range(params["nt"]):
        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)

        if i % params["frameRate"] == 0:  # Save wavefunction data and create a frame
            data.save_wavefunction(psi)
            print( round( abs( params["t"] ) ) )

        params["t"] += params["dt"]  # Increment time count


######################################################Main###################################################################



def main( recalculate:bool=False ) -> None:

    # Generate grid object
    points = (256,256)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)

    # Condensate parameters
    params = {
        "c0": 10,
        "c2": -0.5,
        "p": 0,
        "q": -1,
        "trap": 0.0,
        "n0": 1,
        # Time params
        "dt": (1) * 1e-2,
        "nt": 100_000,
        "t": 0,
        "frameRate": 1000,
        }

    # Generate wavefunction object, set initial state and add noise
    # psi = gpe.SpinOneWavefunction(grid)
    # psi.set_ground_state("antiferromagnetic", params)
    # psi = vortexBoundaryPair( grid, params, 20, 32 )
    # # psi = kelvinHelmholtzBoundary( grid, params, 5 )
    psi = majorityUpVortex( grid, params, 50 )
    # psi.add_noise( "outer", 0, 1e-4 )

    # psi = polarVortexPair( grid, params, 25 )
    psi.add_noise( "all", 0.0, 1e-4 )

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()
    print( time.ctime() )

    # os.makedirs('frames', exist_ok=True)
    # ani.takeFrame( psi, params, 'frames', 0, 'ARG_PLUS_DEBUG', [] )


    chartType = 'mag'
    name = 'majority_up_50'
    movieName = name +  '_' + chartType +'.mp4'
    fileName = name + '.hdf5'
    filePath = "./data/" + fileName
    if not os.path.exists( filePath ) or recalculate:
        # initialEvolution( psi, params, 10000, -1j * 1e-2 )
        # psi.add_noise( 'outer', 0.0, 1e-4 )
        getData( psi, params, fileName, "data" )
        print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

    file = h5py.File( filePath, "r" )
    scalars = hdf5ReadScalars( file )
    waveFunc = file[ "wavefunction" ]
    # vorticity = pseudoVorticity( waveFunc )
    ani.createMovie( waveFunc, scalars, movieName, chartType, ["q","c0","c2","dt"], 'FM' )
        
        


if __name__ == "__main__":
    main()
