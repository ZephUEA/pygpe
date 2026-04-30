import pygpe.spintwo as gpe
from pygpe.spintwo.relaxationPrime import SpinorBECGroundState2D, Spinor
import os
import time
import h5py
import animation as ani
import matplotlib.pyplot as plt
import numpy as np
import random
import polytope as pc
from pygpe.shared.polyhedron import PolyhedronProjector

# np.seterr(all='raise')


pauliX = np.array([[0,1,0,0,0],[1,0,np.sqrt(3/2),0,0],[0,np.sqrt(3/2),0,np.sqrt(3/2),0],[0,0,np.sqrt(3/2),0,1],[0,0,0,1,0]], dtype='complex128')
pauliY = 1j * np.array([[0,-1,0,0,0],[1,0,-np.sqrt(3/2),0,0],[0,np.sqrt(3/2),0,-np.sqrt(3/2),0],[0,0,np.sqrt(3/2),0,-1],[0,0,0,1,0]], dtype='complex128')
pauliZ = np.array([[2,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,-1,0],[0,0,0,0,-2]], dtype='complex128')
paulis = [pauliX,pauliY,pauliZ]


def getData( psi:gpe.SpinTwoWavefunction, params:dict, fileName:str, dataPath:str ) -> None:
    data = gpe.DataManager(fileName, dataPath, psi, params)
    percentages = int(params['nt']/100)
    try:
        for i in range(params["nt"]):
            # if i == 580:
            #     print(i)
            if i % params["frameRate"] == 0 :  # Save wavefunction data and create a frame
                data.save_wavefunction(psi)
            
            if i % percentages ==0 :
                print(f'{i//percentages}% ')
                print( f'Mag = {np.sum(2*(abs(psi.plus2_component)**2 - abs(psi.minus2_component)**2) + abs(psi.plus1_component)**2 - abs(psi.minus1_component)**2)}')
            

            # Evolve wavefunction
            gpe.step_wavefunction(psi, params)

            params["t"] += params["dt"]  # Increment time count
    except FloatingPointError:
        print(f't={params['t']}, i={i}')

def getRelaxation(grid:gpe.Grid, params:dict, psi, fileName, dataPath  ):
    filePath = dataPath + f'/{fileName}'
    createSaveFile( psi, params, filePath )
    system = SpinorBECGroundState2D( grid, params, psi )
    percentages = int(params['nt']/100)
    initials = (sum( [np.sum( abs(system.waveFunction[i])**2) for i in [-2,-1,0,1,2]] ) ,
                sum( [np.sum( i * abs(system.waveFunction[i])**2) for i in [-2,-1,0,1,2]] )   )
    
    for i in range(params["nt"]):
        if i % percentages == 0 :
            print(f'{i//percentages}% ')
            # print( f'N = {sum( [np.sum( abs(system.waveFunctions[-1][i])**2) for i in [-2,-1,0,1,2]] ) }')
            # print( f'MagZ = {sum( [np.sum( i * abs(system.waveFunctions[-1][i])**2) for i in [-2,-1,0,1,2]] ) }')
        
        if i % params["frameRate"] == 0 and i != 0:  # Save wavefunction data and create a frame
            saveWavefunction(system.waveFunction, filePath)

        # Evolve wavefunction
        system.fullStep()

        params["t"] += params["dt"]  # Increment time count

    finals = (sum( [np.sum( abs(system.waveFunction[i])**2) for i in [-2,-1,0,1,2]] ) ,
                sum( [np.sum( i * abs(system.waveFunction[i])**2) for i in [-2,-1,0,1,2]] )   )
    print( rf'$|\Delta N|= {abs(initials[0]-finals[0])} |\Delta M|={abs(initials[1]-finals[1] )}$' )

def createSaveFile(spinor, scalars, filePath):
    with h5py.File(filePath, 'w') as file:
        params_group = file.create_group("parameters")
        wavefunction_group = file.create_group("wavefunction")

        for key in scalars:
            params_group.create_dataset(key, data=scalars[key])

        # Store initial arrays with shape (1, Nx, Ny)
        # maxshape=(None, Nx, Ny) allows unlimited growth along axis 0
        for name, component in [
            ('psi_plus2',  spinor[2]),
            ('psi_plus1',  spinor[1]),
            ('psi_zero',   spinor[0]),
            ('psi_minus1', spinor[-1]),
            ('psi_minus2', spinor[-2]),  # Note: you had spinor[2] here, likely a bug
        ]:
            data = component[np.newaxis, ...]  # shape: (1, Nx, Ny)
            wavefunction_group.create_dataset(
                name,
                data=data,
                dtype="complex128",
                maxshape=(None, *component.shape),  # None = unlimited on axis 0
                chunks=(1, *component.shape),        # one chunk per timeframe
            )

def saveWavefunction(spinor, filePath):
    with h5py.File(filePath, 'a') as file:
        wf = file["wavefunction"]

        components = {
            'psi_plus2':  spinor[2],
            'psi_plus1':  spinor[1],
            'psi_zero':   spinor[0],
            'psi_minus1': spinor[-1],
            'psi_minus2': spinor[-2],
        }

        for name, component in components.items():
            ds = wf[name]
            current_len = ds.shape[0]       # current number of saved frames
            ds.resize(current_len + 1, axis=0)  # grow by 1 along time axis
            ds[current_len] = component     # write new 2D array into new slot

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

def createFilmFromFile(filePath, filmName, frames_dir, filmType='MAG' ):
    file = h5py.File( filePath, 'r')
    psi = file['wavefunction']
    scalars = hdf5ReadScalars( file )

    psiP2 = psi['psi_plus2'][()]
    psiP1 = psi['psi_plus1'][()]
    psi0 = psi['psi_zero'][()]
    psiM1 = psi['psi_minus1'][()]
    psiM2 = psi['psi_minus2'][()]

    xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   
    xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )

    os.makedirs(frames_dir, exist_ok=True)
    for frame in range( psi0.shape[0] ):
        frame_path = f"{frames_dir}/frame_{frame:04d}.png"
        
        match filmType:
            case 'MAG':
                fig, ax = plt.subplots( figsize=(6,6))
                mag = ax.pcolormesh(
                (xMesh),
                (yMesh),
                ( 2*(abs(psiP2[frame,:,:])**2-abs(psiM2[frame,:,:])**2) + abs(psiP1[frame,:,:])**2 - abs(psiM1[frame,:,:])**2 ),
                vmin=-2, vmax=2 )
                ax.set_aspect('equal')
                fig.colorbar( mag )
            case 'A00':
                fig, ax = plt.subplots( figsize=(6,6))
                singlet = ax.pcolormesh(
                (xMesh),
                (yMesh),
                abs( 2*psiP2[frame,:,:]*psiM2[frame,:,:] - 2*psiP1[frame,:,:]*psiM1[frame,:,:] + psi0[frame,:,:]**2 )**2 / 5,
                vmin=0, vmax=0.2 )
                ax.set_aspect('equal')
                fig.colorbar( singlet )
            case 'A30':
                fig, ax = plt.subplots( figsize=(6,6))
                triplet = ax.pcolormesh(
                (xMesh),
                (yMesh),
                abs(3 * np.sqrt(3/2)* (psiP1[frame,:,:]**2 * psiM2[frame,:,:] + psiP2[frame,:,:] * psiM1[frame,:,:]**2 ) + psi0[frame,:,:]
                * (psi0[frame,:,:]**2 - 3 * psiP1[frame,:,:] * psiM1[frame,:,:] - 6 * psiP2[frame,:,:] * psiM2[frame,:,:] ))**2,
                vmin=0, vmax=2 )
                ax.set_aspect('equal')
                fig.colorbar( triplet )
            case 'ABSMAG':
                fig, ax = plt.subplots( figsize=(6,6))
                magZ = 2*(abs(psiP2[frame,:,:])**2-abs(psiM2[frame,:,:])**2) + abs(psiP1[frame,:,:])**2 - abs(psiM1[frame,:,:])**2
                magPlus = (2 * (np.conj(psiP2)*psiP1 + np.conj(psiM1)*psiM2) + np.sqrt(6)*(np.conj(psiP1)*psi0 + np.conj(psi0)*psiM1))[frame,:,:]
                mag = ax.pcolormesh(
                (xMesh),
                (yMesh),
                np.sqrt( abs(magZ)**2 + abs( magPlus * np.conj(magPlus) )),
                vmin=0, vmax=2 )
                ax.set_aspect('equal')
                fig.colorbar( mag )
            case 'ALL':
                fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
                mag = axs[0][0].pcolormesh(
                (xMesh),
                (yMesh),
                ( 2*(abs(psiP2[frame,:,:])**2-abs(psiM2[frame,:,:])**2) + abs(psiP1[frame,:,:])**2 - abs(psiM1[frame,:,:])**2 ),
                vmin=-2, vmax=2 )
                axs[0][0].set_aspect('equal')
                axs[0][0].set_title(r'$M_z$')
                fig.colorbar( mag )

                singlet = axs[1][0].pcolormesh(
                (xMesh),
                (yMesh),
                abs( 2*psiP2[frame,:,:]*psiM2[frame,:,:] - 2*psiP1[frame,:,:]*psiM1[frame,:,:] + psi0[frame,:,:]**2 )**2 / 5,
                vmin=0, vmax=0.2 )
                axs[1][0].set_aspect('equal')
                axs[1][0].set_title(r'$|A_{00}|^2$')
                fig.colorbar( singlet )


                triplet = axs[1][1].pcolormesh(
                (xMesh),
                (yMesh),
                abs(3 * np.sqrt(3/2)* (psiP1[frame,:,:]**2 * psiM2[frame,:,:] + psiP2[frame,:,:] * psiM1[frame,:,:]**2 ) + psi0[frame,:,:]
                * (psi0[frame,:,:]**2 - 3 * psiP1[frame,:,:] * psiM1[frame,:,:] - 6 * psiP2[frame,:,:] * psiM2[frame,:,:] ))**2,
                vmin=0, vmax=2 )
                axs[1][1].set_aspect('equal')
                axs[1][1].set_title(r'$|A_{30}|^2$')
                fig.colorbar( triplet )

                magZ = 2*(abs(psiP2[frame,:,:])**2-abs(psiM2[frame,:,:])**2) + abs(psiP1[frame,:,:])**2 - abs(psiM1[frame,:,:])**2
                magPlus = (2 * (np.conj(psiP2)*psiP1 + np.conj(psiM1)*psiM2) + np.sqrt(6)*(np.conj(psiP1)*psi0 + np.conj(psi0)*psiM1))[frame,:,:]
                magtot = axs[0][1].pcolormesh(
                (xMesh),
                (yMesh),
                np.sqrt( abs(magZ)**2 + abs( magPlus * np.conj(magPlus) )),
                vmin=0, vmax=2 )
                axs[0][1].set_aspect('equal')
                axs[0][1].set_title(r'$|M|$')
                fig.colorbar( magtot )
    


        plt.savefig(frame_path)

        plt.close()
    
    ani.movieFromFrames( filmName, frames_dir )


def circularInfinitePotential( grid, radius, magnitude ):
    x = grid.x_mesh
    y = grid.y_mesh
    r = np.sqrt(abs(x)**2 + abs(y)**2)

    trap = np.zeros(r.shape)
    trap[abs(r)>radius] = magnitude
    return trap

def totalEnergyPlot( psi, scalars ):
    energies = []
    for frame in range( scalars['nt'] // scalars['frameRate'] ):
        psiPlus2 = psi['psi_plus2'][:,:,frame]
        psiPlus1 = psi['psi_plus1'][:,:,frame]
        psiZero = psi['psi_zero'][:,:,frame]
        psiMinus1 = psi['psi_minus1'][:,:,frame]
        psiMinus2 = psi['psi_minus2'][:,:,frame]

        gradPlus2X, gradPlus2Y = np.gradient( psiPlus2, scalars['dx'], scalars['dy'] )
        gradPlus1X, gradPlus1Y = np.gradient( psiPlus1, scalars['dx'], scalars['dy'] )
        gradZeroX, gradZeroY = np.gradient( psiZero, scalars['dx'], scalars['dy'] )
        gradMinus1X, gradMinus1Y = np.gradient( psiMinus1, scalars['dx'], scalars['dy'] )
        gradMinus2X, gradMinus2Y = np.gradient( psiMinus2, scalars['dx'], scalars['dy'] )

        gradEnergy = np.sum( np.conj( gradPlus2X ) * gradPlus2X + np.conj(gradPlus2Y) * gradPlus2Y + 
                             np.conj( gradPlus1X ) * gradPlus1X + np.conj(gradPlus1Y) * gradPlus1Y + 
                             np.conj( gradZeroX ) * gradZeroX + np.conj(gradZeroY) * gradZeroY + 
                             np.conj( gradMinus1X ) * gradMinus1X + np.conj(gradMinus1Y) * gradMinus1Y +
                             np.conj( gradMinus2X ) * gradMinus2X + np.conj(gradMinus2Y) * gradMinus2Y )
        
        densEnergy = np.sum( ( abs(psiPlus2)**2 + abs(psiPlus1)**2 + abs(psiZero)**2 + abs(psiMinus1)**2 + abs(psiMinus2)**2 ) ** 2 )

        magXEnergy = ( np.conj(psiPlus2)*psiPlus1 + np.conj(psiPlus1)*psiPlus2 + np.conj(psiMinus2)*psiMinus1 + np.conj(psiMinus1)*psiMinus2 +
                      np.sqrt(3) * (np.conj(psiZero)*(psiPlus1 + psiMinus1) + np.conj(psiPlus1 + psiMinus1)*psiZero) /2 )
        
        magYenergy = 1j * ( -np.conj(psiPlus2)*psiPlus1 + np.conj(psiPlus1)*psiPlus2 + np.conj(psiMinus2)*psiMinus1 - np.conj(psiMinus1)*psiMinus2 +
                      np.sqrt(3) * (np.conj(psiZero)*(psiPlus1 - psiMinus1) + np.conj(-psiPlus1 + psiMinus1)*psiZero) /2 )
        
        magZEnergy = 2*( abs(psiPlus2)**2 - abs(psiMinus2)**2 ) + abs(psiPlus1)**2 - abs(psiMinus1)**2

        magEnergy = np.sum( abs(magXEnergy) ** 2 + abs(magYenergy) ** 2 + abs(magZEnergy) ** 2 )

        singletEnergy = np.sum( abs( 2 * psiPlus2 * psiMinus2 -2*psiPlus1*psiMinus1 + psiZero**2)**2 )/5

        energies.append( gradEnergy + scalars['c0'] * densEnergy + scalars['c2'] * magEnergy  + scalars['c4'] * singletEnergy )

    ts = np.linspace( 0, scalars['dt']*scalars['nt'], scalars['nt']//scalars['frameRate'] )
    plt.plot( -ts.imag, np.array(energies).real )
    plt.show()




def randomInitial( grid, mag ):
    psi = gpe.SpinTwoWavefunction(grid)
    phases = [ np.exp(1j* 2*np.pi*random.random()) for _ in range(5) ]
    hMatrix = np.array([[1,1,0],[-1,0,0],[1,-2,1],[0,0,-1],[0,1,1]])
    hVector = np.array([(2+mag)/4,0,0,0,(2-mag)/4])
    polyhedron = PolyhedronProjector( pc.Polytope( hMatrix, hVector ) )
    point = polyhedron.project( [random.random(), random.random(), random.random()] )
    initials = [
        np.sqrt(abs((2+mag)/4 - point[0]-point[1]))*phases[0],
        np.sqrt(2*point[0])*phases[1],
        np.sqrt(abs(2*point[1]-point[0]-point[2]))*phases[2],
        np.sqrt(2*point[2])*phases[3],
        np.sqrt(abs((2-mag)/4 - point[2]-point[1]) )*phases[4]
    ]
    psi.set_wavefunction(*initials)
    return psi

    

def main( fileName, mag ):

    power2 = 8
    # Generate grid object
    points = (2**power2, 2**power2)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)

    # trap = infinitePotential( grid, 2**(power2-1) - 2 * (power2-5), 2**(power2-1) - 2 * (power2-5) )
    circularTrap = circularInfinitePotential( grid, 2**(power2-2) - 2*(power2-5), 1e10 )
    # Condensate parameters
    params = {
        "c0": 20,
        "c2": -4,
        "c4": 4,
        "p": 0, # p not yet implemented
        "q": -1,
        "trap": circularTrap,
        "n0": 1,
        # Time params
        "dt": (1) * 1e-2,
        "nt": 1000,
        "t": 0,
        'nx':points[0],
        'ny':points[1],
        'dx':grid_spacings[0],
        'dy':grid_spacings[1],
        "frameRate": 10,
    }
    # psi = gpe.SpinTwoWavefunction(grid)

    # # M >= 0 for relaxationPrime to work
    # psi.set_wavefunction(0,0,1,0,0)

    psi = randomInitial( grid, mag )

    psi.add_noise('all', 0.0, 1e-4)

    psi.plus2_component[params['trap'] != 0] = 0 
    psi.plus1_component[params['trap'] != 0] = 0 
    psi.zero_component[params['trap'] != 0] = 0
    psi.minus1_component[params['trap'] != 0] = 0
    psi.minus2_component[params['trap'] != 0] = 0

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()

    filePath = 'dataSpin2'


    os.makedirs(filePath, exist_ok=True )
    getRelaxation( grid, params,  Spinor( psi.plus2_component, psi.plus1_component, psi.zero_component, psi.minus1_component, psi.minus2_component ), fileName, filePath )
    
    # extractStructure( spinors, params, filePath + f'/{fileName}', frameRate=params['frameRate'] )

    print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

    # film( spinors, params, 'spin2GroundStates/ferroConditionsMag.mp4', 'frames', frameRate=params['frameRate'], filmType='MAG' )



    # if not os.path.exists( filePath ) or recalculate:
    #     getData( psi, params, fileName, targetDirectory )
    

if __name__ == '__main__':
    name = 'ferroConditionsMag1Q-1'
    fileName = name + '.hdf5'
    path = './dataSpin2/' + fileName

    calculate = True
    mag = 1

    if calculate:
        main( fileName, mag )

    file = h5py.File( path, 'r')
    waveFunc = file['wavefunction']
    scalars = hdf5ReadScalars( file )

    createFilmFromFile(f'dataSpin2/{name}.hdf5', f'spin2GroundStates/{name}All.mp4', 'frames', 'ALL')
