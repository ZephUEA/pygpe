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
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def countRegions( waveFunc, scalars, orderFunc, frame:int, threshold=0.7, takePicture=False ) -> int:
    
    def mask( coord, scalars ):
        ( x, y ) = coord
        nx = scalars["nx"]
        ny = scalars["ny"]
        return [((x-1) % nx,y),((x+1) % nx,y),(x,(y-1) % ny),(x,(y+1) % ny)]
    

    orderParam = orderFunc( waveFunc, frame )
    regions = set( map( lambda x: tuple(x), handle_array( cp.argwhere( abs( orderParam ) > threshold ) ) ) )
    if takePicture:
        coords = cp.array(list(regions))
    
        # Extract x and y coordinates
        x = ( coords[:, 0] - scalars['nx']/2 )* scalars['dx'] 
        y = ( coords[:, 1] - scalars['ny']/2 )* scalars['dy']

        return x, y
    count = 0
    while len( regions ) > 0:
        startCoord = regions.pop()
        regions.add( startCoord )
        currentRegion = { startCoord }
        foundCoords = set()
        while len( currentRegion ) > 0:
            currentCoord = currentRegion.pop()
            foundCoords.add( currentCoord )
            regions.remove( currentCoord )
            for item in mask( currentCoord, scalars ):
                if item in regions and item not in currentRegion:
                    currentRegion.add( item )
        if len( foundCoords ) > 100:
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

def getData( psi:gpe.SpinOneWavefunction, params:dict, fileName:str, dataPath:str ) -> None:
    data = gpe.DataManager(fileName, dataPath, psi, params)
    for i in range(params["nt"]):

        if i % params["frameRate"] == 0:  # Save wavefunction data and create a frame
            data.save_wavefunction(psi)
            print( round( params["t"] ) )
        

        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)

        params["t"] += params["dt"]  # Increment time count

        if params["dq"] < 0:
            params["q"] = ( params["Q_0"] - ( params["t"].real / params["tau_q"] ) ) * abs( params["c2"] ) * params["n0"]
            if params["t"] >= params["tau_q"] * 5:
                params["dq"] = 0

def correlatorFM( psi:dict, scalars:dict ) -> cp.ndarray:
    nx = scalars["nx"]
    plusComp = cp.array( psi['psi_plus'][()] )
    minusComp = cp.array( psi['psi_minus'][()] )
    magnetisation = abs( plusComp ) ** 2 - abs( minusComp ) ** 2
    magFFT = cp.fft.fft2( magnetisation, axes=(0,) )
    autoCorrelation = cp.fft.ifft2( abs( magFFT ) ** 2, axes=(0,) ).real
    return autoCorrelation / ( nx )

def chartCorrelator( psi, scalars, scale = False ) -> None:
    zeroTime = 0 # int( scalars["tau_q"] / ( scalars["dt"] * scalars['frameRate'] ) )
    freezingTime = 10 #int( cp.sqrt( scalars["tau_q"] ) )
    correlation = corr.correlatorFM( psi, scalars )
    (xs, averageCorrelation ) = corr.radialAverage( correlation, scalars )
    times = list( filter( lambda x: x < averageCorrelation.shape[ 1 ], [ zeroTime + i for i in range( 0, 6 * freezingTime, freezingTime // 3 ) ] ) )
    firstZeroValues = []
    for t in times:
        if scale:
            endpoint = min( enumerate( averageCorrelation[ :,t ] ), key=lambda x: x[ 1 ] )[ 0 ]
            try:
                firstZero = corr.firstZero( averageCorrelation[ :, t ], endpoint )
                divider = xs[ firstZero ]
            except AssertionError:
                divider = 1
            firstZeroValues.append( divider )
            plt.plot( handle_array( xs / divider ), handle_array( averageCorrelation / averageCorrelation[0,:] )[:,t] )
            plt.xlim( 0, 6 )
            plt.xlabel('Radial Distance x/L(t)')
        else:
            plt.plot( handle_array( xs ), handle_array( averageCorrelation[:,t] ) )
            plt.xlabel('Radial distance x')
    
    
    plt.ylabel('Correlation')
    plt.legend( [ 't= ' + str( ( t - zeroTime ) * ( scalars["dt"] * scalars["frameRate"] ) )  for t in times ] )
    plt.title('Correlation vs normalised radius\nTime since transition: ' + ani.prettyString(scalars, ['Q_0','tau_q', 'p'] ) )
    plt.show()
    # plt.savefig( fileName )
    plt.cla()
    if scale:
        mindex = min( enumerate( firstZeroValues ), key = lambda x: x[1] )[0] + 1
        xs = cp.array( [ ( t - zeroTime ) * ( scalars["dt"] * scalars["frameRate"] ) for t in times[mindex:] ] )
        plt.loglog( xs, firstZeroValues[mindex:] )
        ( b, m ) = corr.bestFitLine( cp.log10(xs), cp.log10(firstZeroValues[mindex:]) )
        plt.loglog( xs, ( 10 ** b ) * ( xs ** m ) )
        plt.legend([ 'data', f'b={b}, m={m}' ])
        plt.ylabel('L(t)')
        plt.xlabel('t - tau')
        plt.title('L(t) vs time since transition')
        plt.show()
        plt.cla()


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
    plt.cla()
    
def countDomains( psi, scalars, givenFrame, threshold=0.7, takePicture=False ):

    def magnetisation( waveFunc, frame ):
        return cp.array( abs( waveFunc["psi_plus"][ :, :, frame ] ) ** 2 - abs( waveFunc["psi_minus"][ :, :, frame ] ) ** 2 )
    
    # Need to find the frame at which the transition has just occured so that we can count domains
    # This should occur approximatly 1 freezing time after the zeroTime.
    return countRegions( psi, scalars, magnetisation, givenFrame, threshold=threshold, takePicture=takePicture )

def maxDomains( psi, scalars ):
    totalFrames = int( scalars["nt"] / scalars['frameRate'] )
    currMax = 0
    for frame in range( totalFrames ):
        domains = countDomains( psi, scalars, givenFrame=frame )
        if currMax > domains:
            # The global maxima is reached monotonically
            return currMax
        currMax = domains

def plotCoursening( psi, scalars, threshold ):
    totalFrames = int( scalars["nt"] / scalars['frameRate'] )
    
    domains = []
    flag = False
    for frame in range( totalFrames ):
        domains.append( countDomains( psi, scalars, givenFrame=frame ) )
        if domains[ -1 ] > threshold:
            flag = True
        if flag == True and domains[ -1 ] < threshold:
            break

    times = cp.linspace( 0, ( scalars["nt"] * scalars["dt"] ), totalFrames )[:len(domains)]
    transitionStartIndex = domains.index( next( filter( lambda x: x != 0, domains ) ) )
    print( transitionStartIndex )
    xs = times[transitionStartIndex:] - scalars["tau_q"]
    plt.loglog( xs, domains[transitionStartIndex:], base=10 )
    plt.title('Coursening')
    plt.xlabel('Time')
    plt.ylabel('Domains')
    bestFitCoeffs = corr.bestFitLine( cp.log10( xs ), cp.log10( domains[transitionStartIndex:] ))
    b, m = bestFitCoeffs
    plt.loglog( xs, ( 10 ** b ) * ( xs ** m )  )
    plt.legend( ['data', f'best fit b={b}, m={m}'] )
    plt.show()

######################################################Main###################################################################



def main( recalculate:bool=False ):
	
    # tau_q = int( os.environ.get('SLURM_ARRAY_TASK_ID') ) ** 2
    tau_q = 1600
    # Generate grid object
    points = 4096
    grid_spacings = 0.5
    grid = gpe.Grid(points, grid_spacings)


    # Condensate parameters
    params = {
        "c0": 1.085,
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

    
    # targetDirectory = "../scratch"
    # fileName = 'kzm_2d_t_'+ str(tau_q) +'.hdf5'
    # filePath = f"./{targetDirectory}/" + fileName

    targetDirectory = "data"
    fileName = 'kzm_1d_t_1600.hdf5'
    filePath = './data/' + fileName

    if not os.path.exists( filePath ) or recalculate:
        getData( psi, params, fileName, targetDirectory )
        print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

    # file = h5py.File( filePath, "r" )
    # scalars = hdf5ReadScalars( file )
    # waveFunc = file[ "wavefunction" ]
    # ani.createFrames( waveFunc, scalars, 'frames', 'GRAD_DENS', ['tau_q'], gridSpacing=grid_spacings[0] )
    
    # if not cupyImport:
    #     chartName = 'correlators/kzm_correlator_instantDebug.png'
    #     chartCorrelator( waveFunc, scalars, chartName, True )

    # chartAtomNumber( waveFunc, scalars, 'potato' )

    # print( countDomains( waveFunc, scalars ) )
    # print( maxDomains( waveFunc, scalars ) )
    # plotCoursening( waveFunc, scalars, 10 )

    # with open( f'./dataFiles/domains{tau_q}.txt', 'a' ) as file:
    #     file.write('\n')
    #     file.write( str( countDomains( waveFunc, scalars ) ) )

if __name__ == "__main__":
    # main()

    file = h5py.File('./data/kzm_1d_t_1600.hdf5', 'r')
    waveFunc = file['wavefunction']
    scalars = hdf5ReadScalars( file )


    fig, ax = plt.subplots( figsize=(12, 12) )
    xs = cp.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ts = cp.arange( 0,scalars['nt'] // scalars['frameRate'] )

    xMesh, tMesh = cp.meshgrid( xs, ts, indexing='ij' )

    magnetisation = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( tMesh ),
                handle_array( abs( waveFunc["psi_plus"][ :, : ] ) ** 2 - abs( waveFunc["psi_minus"][ :, : ] ) ** 2 ), 
                vmin=-1, vmax=1 )
    fig.colorbar( magnetisation )
    plt.show()

    # We want the correlator fucntion
    correlator = correlatorFM( waveFunc, scalars )
    plt.plot( cp.linspace(0,1,correlator.shape[0]//2), correlator[:correlator.shape[0]//2,-1])
    plt.show()