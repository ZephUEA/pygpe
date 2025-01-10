
try:
    import cupy as cp  # type: ignore
    cupyImport = True
except ImportError:
   import numpy as cp
   cupyImport = False
import matplotlib.pyplot as plt

import h5py

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


def main():
    filePaths = [ '../scratch/kzm_2d_t_' + str( num ** 2) + '.hdf5' for num in range(10, 35, 5) ]
    produceCharts( filePaths )

main()
