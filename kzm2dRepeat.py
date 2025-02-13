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
import pygpe.spinone as gpe
import correlation as corr


######################################################Data helper functions###################################################################


def countRegions( waveFunc, scalars, orderFunc) -> int:
    
    def mask( coord, scalars ):
        ( x, y ) = coord
        nx = scalars["nx"]
        ny = scalars["ny"]
        return [((x-1) % nx,y),((x+1) % nx,y),(x,(y-1) % ny),(x,(y+1) % ny)]
    
    
    orderParam = orderFunc( waveFunc )
    regions = set( map( lambda x: tuple(x), handle_array( cp.argwhere( abs( orderParam ) > 0.7 ) ) ) )
    count = 0
    while len( regions ) > 0:
        startCoord = regions.pop()
        regions.add( startCoord )
        currentRegion = { startCoord }
        foundCoords = set()
        while len( currentRegion ) > 0:
            currentCoord = currentRegion.pop()
            regions.remove( currentCoord )
            foundCoords.add( currentCoord )
            for item in mask( currentCoord, scalars ):
                if item in regions and item not in currentRegion:
                    currentRegion.add( item )
        if len( foundCoords ) > 100:
            count += 1
    return count

def correlatorFM( psi:gpe.SpinOneWavefunction, scalars:dict ) -> cp.ndarray:
    nx = scalars["nx"]
    ny = scalars["ny"]
    plusComp = cp.array( psi.plus_component )
    minusComp = cp.array( psi.minus_component )
    magnetisation = abs( plusComp ) ** 2 - abs( minusComp ) ** 2
    magFFT = cp.fft.fft2( magnetisation, axes=(0,1) )
    autoCorrelation = cp.fft.ifft2( abs( magFFT ) ** 2, axes=(0,1) ).real
    return autoCorrelation / ( nx * ny )

######################################################Data generation and visualisation###################################################################

def countDomains( psi, scalars ):

    def magnetisation( waveFunc ):
        return cp.array( abs( waveFunc.plus_component[()] ) ** 2 - abs( waveFunc.minus_component[()] ) ** 2 )
    
    # Need to find the frame at which the transition has just occured so that we can count domains
    # This should occur approximatly 1 freezing time after the zeroTime.

    return countRegions( psi, scalars, magnetisation )

def getData( psi, params:dict, circles:cp.ndarray, threshold:int=6 ) -> tuple:

    domains = []
    times = []
    flag = False
    size = min( params['nx'],params['ny'])//2
    xs = cp.linspace( 0, size * params['dx'], size )
    correlations = []
    for step in range(params["nt"]):

        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)

        params["t"] += params["dt"]  # Increment time count
        
        if params["dq"] < 0:
            params["q"] = ( params["Q_0"] - ( params["t"].real / params["tau_q"] ) ) * abs( params["c2"] ) * params["n0"]
            if params["t"] >= params["tau_q"] * ( int( params['Q_0'] * 2 ) + 1 ): #This is to stop at q=-0.5 
                params["dq"] = 0

        if step % params["frameRate"] == 0:       
            domains.append( countDomains( psi, params ) ) 
            times.append( params["t"] )
            correlations.append( cp.sum( correlatorFM( psi, params )[:,:,None] * circles, axis=(0,1) ) )
            if domains[ -1 ] > threshold:
                flag = True
            if flag == True and domains[ -1 ] < threshold:
                break

    return ( cp.array(times) - params["tau_q"], cp.array( domains ), xs, cp.array( correlations ) )


######################################################Main###################################################################



def main():
	
    tau_q = int( os.environ.get('SLURM_ARRAY_TASK_ID') ) ** 2
    #tau_q = 0
    # Generate grid object
    nx = 2048
    ny = 2048
    points = (nx,ny)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)
    circles = corr.createCircles( ( nx, ny ) )
    # Condensate parameters
   
    for i in range( 10 ):
        params = {
            "c0": 4.34, # 4.34 for Li7 and 432 for Rb87
            "c2": -0.5,
            "p": 0,
            "q": 0.5,
            "n0": 1,
            "trap":0.0,
            # Time params
            "dt": (1) * 1e-2,
            "dq": -1,
            "nt": 1_000_000,
            "t": 0,
            "tau_q":tau_q,
            "frameRate":1000,
            "Q_0" : 1,
            "nx" : nx,
            "ny" : ny,
            "dx" : grid_spacings[0],
            "dy" : grid_spacings[1]
        }
        psi = gpe.SpinOneWavefunction(grid)
        psi.set_ground_state("BA", params)
        psi.add_noise("all", 0.0, 1e-4)

        if tau_q == 0:
            params['dq'] = 0
            params['q'] = -0.5        

        psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
        start_time = time.time()
        results = getData( psi, params, circles )
        print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

        with h5py.File( f'./../scratch/baLiDomains{tau_q}.hdf5', 'a' ) as file:
            grp = file.create_group( f'run{i}' )
            grp.create_dataset( 'Times',data=handle_array( results[0] ) )
            grp.create_dataset( 'Domains', data=handle_array( results[1] ) )
            grp.create_dataset( 'Radii', data=handle_array( results[2] ) )
            grp.create_dataset( 'Correlations', data=handle_array( results[3] ) )

if __name__ == "__main__":
    main()
