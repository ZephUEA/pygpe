import time
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Applications/ffmpeg"

try:
    import cupy as cp  # type: ignore
    cupyImport = True
except ImportError:
   import numpy as cp
   cupyImport = False

from pygpe.shared.utils import handle_array
import pygpe.spinone as gpe



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
        while len( currentRegion ) > 0:
            currentCoord = currentRegion.pop()
            regions.remove( currentCoord )
            for item in mask( currentCoord, scalars ):
                if item in regions and item not in currentRegion:
                    currentRegion.add( item )

        count += 1
    return count
        

######################################################Data generation and visualisation###################################################################

def countDomains( psi, scalars ):

    def magnetisation( waveFunc ):
        return cp.array( abs( waveFunc.plus_component[()] ) ** 2 - abs( waveFunc.minus_component[()] ) ** 2 )
    
    # Need to find the frame at which the transition has just occured so that we can count domains
    # This should occur approximatly 1 freezing time after the zeroTime.

    return countRegions( psi, scalars, magnetisation )

def getData( psi, params:dict ) -> None:

    maxDomains = 0

    for _ in range(params["nt"]):

        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)

        params["t"] += params["dt"]  # Increment time count
        params["q"] = ( params["Q_0"] - ( params["t"] / params["tau_q"] ) ) * abs( params["c2"] ) * params["n0"]
        domains = countDomains( psi, params )
        if domains > maxDomains:
            maxDomains = domains
        if domains < maxDomains // 10:
            break

    return maxDomains
    

######################################################Main###################################################################



def main( ):
	
    tau_q = 900
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
        "n0": 1,
        "trap":0.0,
        # Time params
        "dt": (1) * 1e-2,
        "dq": -1,
        "nt": 200_000,
        "t": 0,
        "tau_q":tau_q,
        "Q_0" : 1.0,
        "nx" : 512,
        "ny" : 512
    }

    # Generate wavefunction object, set initial state and add noise
    psi = gpe.SpinOneWavefunction(grid)
    psi.set_ground_state("BA", params)
    psi.add_noise("all", 0.0, 1e-4)

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()

    
    domains = getData( psi, params )
    print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')


    with open( f'./dataFiles/domains{tau_q}.txt', 'a' ) as file:
        file.write('\n')
        file.write( str( domains ) )

if __name__ == "__main__":
    for _ in range( 50 ):
        main()
