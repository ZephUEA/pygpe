import pygpe.spinone as gpe
import os
import time
import h5py
import animation as ani
import numpy as np

# np.seterr(all='raise')


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



def getData( psi:gpe.SpinOneWavefunction, params:dict, fileName:str, dataPath:str ) -> None:
    data = gpe.DataManager(fileName, dataPath, psi, params)
    percentages = int(params['nt']/100)
    for i in range(params["nt"]):

        if i % params["frameRate"] == 0:  # Save wavefunction data and create a frame
            data.save_wavefunction(psi)
        
        if i % percentages == 0 :
            print(f'{i//percentages}% ')
            if i // percentages == 50:
                params['qSpace'] = 0
        
        
        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)

        params["t"] += params["dt"]  # Increment time count



def main( recalculate:bool=False ):


    # Generate grid object
    points = (128,128)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)
    
    qArray = np.concat( (np.zeros((points[0],points[1]//2)),np.ones((points[0],points[1]-points[1]//2))), axis=1 )

    # Condensate parameters
    params = {
        "c0": 20, # Sodium params, need to ask Magnus
        "c2": 0.5,
        "p": 0,
        'q': 0,
        "qSpace": qArray,
        "trap": 0,
        "n0": 1,
        # Time params
        "dt": (2) * 1e-3,
        "nt": 1_000_000,
        "t": 0,
        "frameRate": 1000,
    }

  
    psi = gpe.SpinOneWavefunction(grid)
    psi.set_wavefunction( plus_component= 1 - qArray, zero_component=qArray)
    psi.add_noise("all", 0.0, 1e-4)

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()

    targetDirectory = "dataSpin1"
    fileName = 'spatialQCollapse.hdf5'
    filePath = './dataSpin1/' + fileName

    if not os.path.exists( filePath ) or recalculate:
        getData( psi, params, fileName, targetDirectory )
        print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

if __name__ == '__main__':
    main(False)

    file = h5py.File( './dataSpin1/spatialQCollapse.hdf5', 'r')
    waveFunc = file['wavefunction']
    scalars = hdf5ReadScalars( file )

    os.makedirs('frames', exist_ok=True)
    for frame in range(scalars["nt"]//scalars["frameRate"]):
        # ani.takeFrame( waveFunc, scalars, 'frames', frame, 'MAG',[] )
        # ani.allComponentFrame(waveFunc, scalars, frame, 'frames')
        ani.magnetisationQuiverFrame( waveFunc, scalars, frame, 'frames' )
        # ani.superfluidVelocitiesFrame(waveFunc, scalars, frame, 'frames')
        pass
    
    ani.movieFromFrames( 'spatialQCollapseQuiver.mp4', 'frames' )