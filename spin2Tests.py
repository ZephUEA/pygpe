import pygpe.spintwo as gpe
import os
import time
import h5py
import animation as ani
import matplotlib.pyplot as plt
import numpy as np
import pygpe.shared.vortices as vort

# np.seterr(all='raise')

def getData( psi:gpe.SpinTwoWavefunction, params:dict, fileName:str, dataPath:str ) -> None:
    data = gpe.DataManager(fileName, dataPath, psi, params)
    percentages = int(params['nt']/100)
    try:
        for i in range(params["nt"]):
            # if i == 580:
            #     print(i)
            if i % params["frameRate"] == 0:  # Save wavefunction data and create a frame
                data.save_wavefunction(psi)
            
            if i % percentages ==0 :
                print(f'{i//percentages}% ')
            

            # Evolve wavefunction
            gpe.step_wavefunction(psi, params)

            params["t"] += params["dt"]  # Increment time count
    except FloatingPointError:
        print(f't={params['t']}, i={i}')

def sphericalToCartesian( r, theta, phi ):
    x = r * np.sin(theta)*np.cos(phi)
    y = r * np.sin(theta)*np.sin(phi)
    z = r * np.cos(theta) 
    return (x,y,z)

def sphericalRep(spinor):
    pi = np.pi
    phi = np.linspace( 0, 2*pi, 1000 )
    theta = np.linspace( 0, pi, 1000 )
    phi, theta = np.meshgrid(phi, theta)
    y2_2 = np.sqrt( 15/(2*pi) )/4 * np.sin(theta) ** 2 * np.exp( 2 * 1j * phi )
    y2_1 = np.sqrt( 15/(2*pi) )/-2 * np.sin(theta) * np.cos( theta ) * np.exp( 1j * phi )
    y2_0 = np.sqrt( 5/pi )/4 * ( 3 * np.cos(theta) ** 2 - 1 )
    y2_1m = np.sqrt( 15/(2*pi) )/2 * np.sin(theta) * np.cos( theta ) * np.exp( -1j * phi )
    y2_2m = np.sqrt( 15/(2*pi) )/4 * np.sin(theta) ** 2 * np.exp( -2 * 1j * phi )
    harmonicRep = spinor[0] * y2_2 + spinor[1] * y2_1 + spinor[2] * y2_0 + spinor[3] * y2_1m + spinor[4] * y2_2m
    return harmonicRep, theta, phi

def plotHarmonic( spinor ):
    (harmonicRep,theta,phi) = sphericalRep( spinor )
    (x,y,z) = sphericalToCartesian( abs(harmonicRep), theta, phi )
    fig = plt.figure()
    ax = fig.add_subplot( 111 , projection='3d')
    ax.set_aspect('equal')
    norm = plt.Normalize(-np.pi,  np.pi)
    cmap = plt.cm.viridis
    colours = plt.cm.viridis(norm(np.angle(harmonicRep)))
    ax.plot_surface(x, y, z, 
                    linewidth = 0.5, 
                    facecolors = colours, 
                    edgecolor = 'k',
                    antialiased= True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need this line for the colorbar to work
    
    # Add the colorbar
    cbar = fig.colorbar(sm, ax=ax, label='Phase (radians)')
    plt.show()


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

def normalisation(spinor):
    return abs(spinor[0])**2+abs(spinor[1])**2+abs(spinor[2])**2+abs(spinor[3])**2+abs(spinor[4])**2



def magnetisation(spinor):
    return 2*(abs(spinor[0])**2-abs(spinor[4])**2) + abs(spinor[1])**2 - abs(spinor[3])**2

def spinSinglet(spinor):
    return (2*spinor[0]*spinor[4]-2*spinor[1]*spinor[3]+spinor[2]**2)/np.sqrt(5)

def spinTriplet(spinor):
    return 3*np.sqrt(6)*(spinor[4]*spinor[1]**2+spinor[0]*spinor[3]**2)/2 + spinor[2]*(spinor[2]**2 -3*spinor[1]*spinor[3]-6*spinor[0]*spinor[4])

def integratedMagnetisation( spinorGrid ):
    return np.sum( magnetisation(spinorGrid),axis=None) / np.prod(spinorGrid.shape[1:])

def integragedDensity(spinorGrid):
    return np.sum( normalisation(spinorGrid), axis=None) /  np.prod(spinorGrid.shape[1:])

def integragedSpinTriplet(spinorGrid):
    return np.sum(abs( spinTriplet(spinorGrid) )**2,axis=None) /  np.prod(spinorGrid.shape[1:])

def integratedSpinSinglet( spinorGrid ):
    return np.sum(abs(spinSinglet(spinorGrid))**2,axis=None) /  np.prod(spinorGrid.shape[1:])


def sigmoid( x, radius=1 ):
    return ( 2 / ( np.exp(-x/radius) + 1 ) ) - 1


def knotProfile(grid, position):
    y =  (grid.y_mesh - position[1])
    x =  (grid.x_mesh - position[0])
    radius = np.sqrt(x**2 + y**2)

    return sigmoid( radius, 0.1  )

def knotVortexPair( grid, coord1, coord2 ):
    phase1 = vort._calculate_vortex_contribution(grid, coord1[0], coord1[1], 1)
    phase2 = vort._calculate_vortex_contribution(grid, coord2[0], coord2[1], -1)
    phase = phase1 + phase2
    up2Vort1 = knotProfile( grid, coord1 )
    up2Vort2 = knotProfile( grid, coord2 )
    up2 = up2Vort1 * up2Vort2
    down1 = np.sqrt(1-abs(up2)**2)
    psi = gpe.SpinTwoWavefunction(grid)
    psi.set_wavefunction(up2,None,None,down1,None)
    psi.apply_phase(phase,'plus2') # Apply phase twice to doubly wind the vortices
    # psi.apply_phase(phase,'plus2')
    return psi

def harmonicTrap( grid, params, freq=1 ):
    y2 = abs(grid.y_mesh)**2
    x2 = abs(grid.x_mesh)**2
    newFreq = freq/(2*abs(params['c2']*params['n0']))
    trap = ((newFreq**2)*x2 + (newFreq**2)*y2)/2
    spinorComponent = np.exp( -newFreq*(x2 + y2 )/2)
    return trap, spinorComponent


def main( recalculate:bool=False ):


    # Generate grid object
    points = (128,128)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)
    
    # Condensate parameters
    params = {
        "c0": 20,
        "c2": 4,
        "c4": -4,
        "p": 0,
        "q": -0.2,
        "trap": 0,
        "n0": 1,
        # Time params
        "dt": (3) * 1e-4,
        "nt": 100_000,
        "t": 0,
        "frameRate": 1000,
    }

    # trap, spinor = harmonicTrap( grid, params )

    # params['trap'] = trap
    # Generate wavefunction object, set initial state and add noise
    psi = gpe.SpinTwoWavefunction(grid)

    # Want to set two vorticies up in say FM2 phase


    # psi = knotVortexPair( grid, (-2,0),(2,0))
    # psi.set_wavefunction(1/np.sqrt(2),0,0,0,1/np.sqrt(2))
    psi.set_wavefunction(0,1/np.sqrt(2),0,1/np.sqrt(2),0)
    psi.add_noise("all", 0.0, 1e-4)

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()

    
    # targetDirectory = "../scratch"
    # fileName = 'kzm_2d_t_'+ str(tau_q) +'.hdf5'
    # filePath = f"./{targetDirectory}/" + fileName

    targetDirectory = "dataSpin2"
    fileName = 'cyclicStability.hdf5'
    filePath = './dataSpin2/' + fileName

    if not os.path.exists( filePath ) or recalculate:
        getData( psi, params, fileName, targetDirectory )
        print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

if __name__ == '__main__':
    main(True)

    file = h5py.File( './dataSpin2/cyclicStability.hdf5', 'r')
    waveFunc = file['wavefunction']
    scalars = hdf5ReadScalars( file )
    initialGrid = np.array([
            waveFunc['psi_plus2'][:,:,0],
            waveFunc['psi_plus1'][:,:,0],
            waveFunc['psi_zero'][:,:,0],
            waveFunc['psi_minus1'][:,:,0],
            waveFunc['psi_minus2'][:,:,0]
        ])
    print(f'Norm:{integragedDensity(initialGrid)}\nMAGZ:{integratedMagnetisation(initialGrid)}\nA20:{(integratedSpinSinglet(initialGrid))}\nA30:{(integragedSpinTriplet(initialGrid))}')
    
    finalGrid = np.array([
            waveFunc['psi_plus2'][:,:,-1],
            waveFunc['psi_plus1'][:,:,-1],
            waveFunc['psi_zero'][:,:,-1],
            waveFunc['psi_minus1'][:,:,-1],
            waveFunc['psi_minus2'][:,:,-1]
        ])
    print(f'Norm:{integragedDensity(finalGrid)}\nMAGZ:{integratedMagnetisation(finalGrid)}\nA20:{(integratedSpinSinglet(finalGrid))}\nA30:{abs(integragedSpinTriplet(finalGrid))}')
    

    ani.createFrames( waveFunc, scalars, 'frames', 'A30', ['tau_q'], spin=2, gridSpacing=0.5 )
    ani.movieFromFrames( 'trialRunMAG.mp4', 'frames' )