import time
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Applications/ffmpeg"
import h5py

try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
from pygpe.shared.utils import handle_array
from pygpe.shared.vortices import add_dipole_pair
import matplotlib.pyplot as plt

import pygpe.spinone as gpe

import correlation as corr
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

def phaseBoundary( grid, params ):
    psi = gpe.SpinOneWavefunction( grid )
    shape = grid.shape
    phaseWidth = shape[0] // 2
    phaseHeight = shape[1]
    upPhase = cp.concatenate( ( cp.zeros( ( phaseWidth, phaseHeight ), dtype="complex128" ), cp.ones( ( phaseWidth, phaseHeight ), dtype="complex128" ) ) ) * cp.sqrt( params[ "n0" ] )
    downPhase = cp.concatenate( ( cp.ones( ( phaseWidth, phaseHeight ), dtype="complex128" ), cp.zeros( ( phaseWidth, phaseHeight ), dtype="complex128" ) ) ) * cp.sqrt( params[ "n0" ] )
    psi.set_wavefunction(plus_component=upPhase, minus_component=downPhase)
    return psi 

def vortexBoundaryPair( grid, params, threshold ):
    psi = phaseBoundary( grid, params )
    phase = add_dipole_pair( grid, threshold )
    psi.apply_phase( phase, [ "plus", "minus"] )
    psi.add_noise( "all", 0, 1e-4 )
    return psi


######################################################Data generation and visualisation###################################################################

def getData( psi, params:dict, fileName:str, dataPath:str ) -> None:
    data = gpe.DataManager(fileName, dataPath, psi, params)
    for i in range(params["nt"]):

        if i % params["frameRate"] == 0:  # Save wavefunction data and create a frame
            data.save_wavefunction(psi)
            print( round( params["t"] ) )
        
        if i == 0:
            params["p"] = 0

        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)

        params["t"] += params["dt"]  # Increment time count
        params["q"] = ( params["Q_0"] - ( params["t"].real / params["tau_q"] ) ) * abs( params["c2"] ) * params["n0"]


def chartCorrelator( psi, scalars, fileName ) -> None:
    correlation = corr.correlatorFM( psi, scalars )
    averageCorrelation = corr.radialAverage( correlation, scalars )
    times = [1,60,100,200,400]
    for time in times:
        plt.plot( averageCorrelation[:,time] )
    plt.legend([ 't=' + str( time ) + '0' for time in times])
    plt.title( ani.prettyString(scalars, ['Q_0','tau_q', 'p'] ) )
    plt.savefig( fileName )


######################################################Main###################################################################



def main( recalculate:bool=False):

    tau_q = int( os.environ.get('SLURM_ARRAY_TASK_ID') )
    # Generate grid object
    points = (512,512)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)


    # Condensate parameters
    params = {
        "c0": 10,
        "c2": -0.5,
        "p": 0,
        "q": 0.5,
        "trap": 0.0,
        "n0": 1,
        # Time params
        "dt": (1) * 1e-2,
        "dq": -1, # is q increasing or decreasing?
        "nt": 600_000,
        "t": 0,
        "tau_q":tau_q,
        "frameRate": 1000,
        "Q_0" : 1.0
    }

    # Generate wavefunction object, set initial state and add noise
    psi = gpe.SpinOneWavefunction(grid)
    psi.set_ground_state("BA", params)
    psi.add_noise("all", 0.0, 1e-4)

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()

    
    movieName = 'ba_to_fm_p0t500_mag.mp4'
    fileName = 'ba_to_fm_p0t500.hdf5'
    filePath = "./data/" + fileName
    if not os.path.exists( filePath ) or recalculate:
        getData( psi, params, fileName, "data" )
        print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

    file = h5py.File( filePath, "r" )
    scalars = hdf5ReadScalars( file )
    waveFunc = file[ "wavefunction" ]
    # vorticity = corr.pseudoVorticity( waveFunc )
    # createMovie( waveFunc, scalars, movieName, 'MAG' )

    chartName = 'correlators/kzm_correlator_t' + str( tau_q ) + '.png'
    chartCorrelator( waveFunc, scalars, chartName )
        
        


if __name__ == "__main__":
    main()
