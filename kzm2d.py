import time
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Applications/ffmpeg"
import h5py

try:
    import cupy as cp  # type: ignore
    cupyImport = True
except ImportError:
   import numpy as cp
   cupyImport = False

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

def firstZero( xs:cp.ndarray, ys:cp.ndarray ) -> float:
    '''Finds the x value where y first crosses 0, 
    xs and ys should have equal length'''
    assert len( xs ) == len( ys )
    for index in range( len( ys ) ):
        if ys[ index ] * ys[ index + 1 ] < 0:
            return xs[ index ]
    return None


def countRegions( waveFunc, scalars, orderFunc, frame:int ) -> int:
    
    def mask( coord, scalars ):
        ( x, y ) = coord
        nx = scalars["nx"]
        ny = scalars["ny"]
        return [((x-1) % nx,y),((x+1) % nx,y),(x,(y-1) % ny),(x,(y+1) % ny)]
    
    
    orderParam = orderFunc( waveFunc, frame )
    regions = set( map( lambda x: tuple(x), handle_array( cp.argwhere( abs( orderParam ) > 0.7 ) ) ) )
    count = 0
    while len( regions ) > 0:
        startCoord = regions.pop()
        regions.add( startCoord )
        currentRegion = { startCoord }
        while len( currentRegion ) > 0:
            currentCoord = currentRegion.pop()
            regions.remove( currentCoord )
            for item in mask( currentCoord, scalars ):
                if item in regions and item not in currentRegion:
                    currentRegion.add( item )

        count += 1
    return count
        




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
        

        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)

        params["t"] += params["dt"]  # Increment time count
        params["q"] = ( params["Q_0"] - ( params["t"].real / params["tau_q"] ) ) * abs( params["c2"] ) * params["n0"]


def chartCorrelator( psi, scalars, fileName ) -> None:
    zeroTime = int( scalars["tau_q"] / ( scalars["dt"] * scalars['frameRate'] ) )
    freezingTime = int( cp.sqrt( scalars["tau_q"] ) )
    correlation = corr.correlatorFM( psi, scalars )
    ( xs, averageCorrelation ) = corr.radialAverage( correlation, scalars )
    times = list( filter( lambda x: x < averageCorrelation.shape[ 1 ], [ zeroTime + i for i in range( 0, 2 * freezingTime , freezingTime // 5 ) ] ) )
    for time in times:
        plt.plot( handle_array( xs ), handle_array( averageCorrelation[:,time] ) )
    plt.xlabel('Radial Distance')
    plt.ylabel('Correlation')
    plt.legend( [ 't= ' + str( ( time - zeroTime ) * ( scalars["dt"] * scalars["frameRate"] ) )  for time in times ] )
    plt.title('Time since transition \nWith:'  + ani.prettyString(scalars, ['Q_0','tau_q', 'p'] ) )
    plt.savefig( fileName )

def chartAtomNumber( psi, scalars, fileName ) -> None:
    zeroTime = scalars["tau_q"]
    times =  cp.linspace( 0, scalars["nt"] * scalars["dt"], scalars["nt"] // scalars["frameRate"] ) 
    analytic = ( scalars["Q_0"] - ( times ) / scalars["tau_q"] ) / 4 + 0.5
    normalisedTimes = list( filter( lambda x: x < 1, ( times - zeroTime ) / scalars["tau_q"] ) )
    plt.plot( normalisedTimes,  analytic[:len(normalisedTimes)] )
    shape = psi["psi_zero"][()].shape
    normalisedNumberDensity = cp.sum( abs( psi["psi_zero"][()] ) ** 2, axis=( 0, 1 ) )[:len(normalisedTimes)] / ( shape[0] * shape[1] )
    plt.plot( normalisedTimes, normalisedNumberDensity )
    plt.xlabel( 't/tau_Q' )
    plt.ylabel( '0-Comp atom No.' )
    plt.title( 'Atom Number' )
    plt.legend(['Analytic', f'tau_q={scalars["tau_q"]}'])
    plt.show()
    
def countDomains( psi, scalars, givenFrame=None ):

    def magnetisation( waveFunc, frame ):
        return cp.array( abs( waveFunc["psi_plus"][ :, :, frame ] ) ** 2 - abs( waveFunc["psi_minus"][ :, :, frame ] ) ** 2 )
    
    # Need to find the frame at which the transition has just occured so that we can count domains
    # This should occur approximatly 1 freezing time after the zeroTime.
    if givenFrame is None:
        frame = int( ( scalars["tau_q"] + 12 * cp.sqrt(scalars['tau_q']) ) / ( scalars['dt'] * scalars['frameRate' ] ) )
    else:
        frame = givenFrame
    return countRegions( psi, scalars, magnetisation, frame )

def maxDomains( psi, scalars ):
    totalFrames = int( scalars["nt"] / scalars['frameRate'] )
    currMax = 0
    for frame in range( totalFrames ):
        domains = countDomains( psi, scalars, givenFrame=frame )
        if currMax > domains:
            # The global maxima is reached monotonically
            return currMax
        currMax = domains

def plotCoursening( psi, scalars ):
    totalFrames = int( scalars["nt"] / scalars['frameRate'] )
    times = cp.linspace( 0, ( scalars["nt"] * scalars["dt"] ), totalFrames )
    domains = []
    for frame in range(totalFrames):
        domains.append( countDomains( psi, scalars, givenFrame=frame ) )
    plt.plot( times - scalars["tau_q"], domains )
    plt.title('Coursening')
    plt.xlabel('Time')
    plt.ylabel('Domains')
    plt.show()


######################################################Main###################################################################



def main( recalculate:bool=False ):
	
    #tau_q = int( os.environ.get('SLURM_ARRAY_TASK_ID') ) ** 2

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
        "dq": -1,
        "nt": 600_000,
        "t": 0,
        "tau_q":900,
        "frameRate": 1000,
        "Q_0" : 1.0
    }

    # Generate wavefunction object, set initial state and add noise
    psi = gpe.SpinOneWavefunction(grid)
    psi.set_ground_state("BA", params)
    psi.add_noise("all", 0.0, 1e-4)

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()

    
    # targetDirectory = "../scratch"
    # fileName = 'kzm_2d_t_'+ str(tau_q) +'.hdf5'
    # filePath = f"./{targetDirectory}/" + fileName

    targetDirectory = "data"
    fileName = 'kzm_2d_t_900.hdf5'
    filePath = './data/' + fileName

    if not os.path.exists( filePath ) or recalculate:
        getData( psi, params, fileName, targetDirectory )
        print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

    file = h5py.File( filePath, "r" )
    scalars = hdf5ReadScalars( file )
    waveFunc = file[ "wavefunction" ]
    # vorticity = corr.pseudoVorticity( waveFunc )
    # ani.createFrames( waveFunc, scalars, 'frames', 'MAG', ['tau_q'] )
    
    # if not cupyImport:
    #     chartName = 'correlators/kzm_correlator_t900.png'
    #     chartCorrelator( waveFunc, scalars, chartName )

    # chartAtomNumber( waveFunc, scalars, 'potato' )
    print( countDomains( waveFunc, scalars ) )
    print( maxDomains( waveFunc, scalars ) )
    plotCoursening( waveFunc, scalars )

    # with open( f'./dataFiles/domains{tau_q}.txt', 'a' ) as file:
    #     file.write('\n')
    #     file.write( str( countDomains( waveFunc, scalars ) ) )

if __name__ == "__main__":
    main()
