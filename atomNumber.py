
try:
    import cupy as cp  # type: ignore
    cupyImport = True
except ImportError:
   import numpy as cp
   cupyImport = False
import matplotlib.pyplot as plt

import h5py

from pygpe.shared.utils import handle_array

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
    plt.savefig( fileName + '.png' )


def produceCharts( fileNames ) -> None:

    legend = []
    for filename in fileNames:
        file = h5py.File( filename, "r" )
        psi = file["wavefunction"]
        scalars = hdf5ReadScalars( file )
        zeroTime = scalars["tau_q"]
        times =  cp.linspace( 0, scalars["nt"] * scalars["dt"], scalars["nt"] // scalars["frameRate"] ) 
        analytic = ( scalars["Q_0"] - ( times ) / scalars["tau_q"] ) / 4 + 0.5
        normalisedTimes = list( filter( lambda x: x < 1, ( times - zeroTime ) / scalars["tau_q"] ) )
        plt.plot( normalisedTimes,  analytic[:len(normalisedTimes)] )
        shape = psi["psi_zero"][()].shape
        normalisedNumberDensity = cp.sum( abs( psi["psi_zero"][()] ) ** 2, axis=( 0, 1 ) )[:len(normalisedTimes)] / ( shape[0] * shape[1] )
        plt.plot( normalisedTimes, normalisedNumberDensity )
        legend += ['Analytic', f'tau_q={scalars["tau_q"]}']

    plt.xlabel( 't/tau_Q' )
    plt.ylabel( '0-Comp atom No.' )
    plt.title( 'Atom Number' )
    plt.legend( legend )
    plt.savefig( './charts/AtomNumber.png' )


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


def plotCoursening( fileNames ):
    legend = []
    for fileName in fileNames:
        file = h5py.File( fileName, 'r' )
        psi = file['wavefunction']
        scalars = hdf5ReadScalars( file )
        totalFrames = int( scalars["nt"] / scalars['frameRate'] )
        times = cp.linspace( 0, ( scalars["nt"] * scalars["dt"] ), totalFrames )
        domains = []
        for frame in range(totalFrames):
            domains.append( countDomains( psi, scalars, givenFrame=frame ) )
        plt.plot( times - scalars["tau_q"], domains )
        legend.append(f'tau_q={scalars["tau_q"]}') 
    plt.title('Coursening')
    plt.xlabel('Time')
    plt.ylabel('Domains')
    plt.legend( legend )
    plt.savefig( './charts/coursening.png' )

def main():
    filePaths = [ '../scratch/kzm_2d_t_' + str( num ** 2) + '.hdf5' for num in range(10, 35, 5) ]
    produceCharts( filePaths )

main()
