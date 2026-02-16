import pygpe.spintwo as gpe
from pygpe.spintwo.relaxation import SpinorBECGroundState2D, Spinor
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
                print( f'Mag = {np.sum(2*(abs(psi.plus2_component)**2 - abs(psi.minus2_component)**2) + abs(psi.plus1_component)**2 - abs(psi.minus1_component)**2)}')
            

            # Evolve wavefunction
            gpe.step_wavefunction(psi, params)

            params["t"] += params["dt"]  # Increment time count
    except FloatingPointError:
        print(f't={params['t']}, i={i}')

def getRelaxation(grid:gpe.Grid, params:dict, psi  ):
    system = SpinorBECGroundState2D( grid, params, psi )
    percentages = int(params['nt']/100)
    initials = (np.sum(abs(system.waveFunctions[-1][1])**2 +abs(system.waveFunctions[-1][0])**2+ abs(system.waveFunctions[-1][-1])**2 ),
                np.sum(abs(system.waveFunctions[-1][1])**2 - abs(system.waveFunctions[-1][-1])**2 )  )
    
    for i in range(params["nt"]):
        if i % percentages == 0 :
            print(f'{i//percentages}% ')
            print( f'N = {np.sum(abs(system.waveFunctions[-1][1])**2 +abs(system.waveFunctions[-1][0])**2+ abs(system.waveFunctions[-1][-1])**2 )}')
            print( f'MagZ = {np.sum(abs(system.waveFunctions[-1][1])**2 - abs(system.waveFunctions[-1][-1])**2 )}')
        

        # Evolve wavefunction
        system.fullStep()

        params["t"] += params["dt"]  # Increment time count

    finals = (np.sum(abs(system.waveFunctions[-1][1])**2 +abs(system.waveFunctions[-1][0])**2+ abs(system.waveFunctions[-1][-1])**2 ),
                np.sum(abs(system.waveFunctions[-1][1])**2 - abs(system.waveFunctions[-1][-1])**2 )  )
    print( rf'$|\Delta N|= {abs(initials[0]-finals[0])} |\Delta M|={abs(initials[1]-finals[1] )}$' )
    return ( system.waveFunctions,  system.computeChemicalPotentials() )


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


def extractStructure(spinors, scalars, filePath, frameRate=1):
    cut = slice(0,None,frameRate)
    saveSpinors = spinors[cut]
    spinorPlus2 = np.array(list(map( lambda x: x[2], saveSpinors ) ) )
    spinorPlus1 = np.array(list(map( lambda x: x[1], saveSpinors ) ) )
    spinorZero = np.array(list(map( lambda x: x[0], saveSpinors ) ) )
    spinorMinus1 = np.array( list(map( lambda x: x[-1], saveSpinors ) ) )
    spinorMinus2 = np.array( list(map( lambda x: x[-2], saveSpinors ) ) )

    with h5py.File(filePath, 'w') as file:
        # Create the groups first
        params_group = file.create_group("parameters")
        wavefunction_group = file.create_group("wavefunction")
        
        # Add scalar parameters
        for key in scalars:
            params_group.create_dataset(key, data=scalars[key])
        
        # Create and populate wavefunction datasets
        wavefunction_group.create_dataset(
            'psi_plus2',
            data=spinorPlus2,
            dtype="complex128",
        )
        wavefunction_group.create_dataset(
            'psi_plus1',
            data=spinorPlus1,
            dtype="complex128",
        )
        wavefunction_group.create_dataset(
            'psi_zero',
            data=spinorZero,
            dtype="complex128",
        )
        wavefunction_group.create_dataset(
            'psi_minus1',
            data=spinorMinus1,
            dtype="complex128",
        )
        wavefunction_group.create_dataset(
            'psi_minus2',
            data=spinorMinus2,
            dtype="complex128",
        )


def film( spinors, scalars, filmName, frames_dir, frameRate=1, filmType='MAG' ):
    os.makedirs(frames_dir, exist_ok=True)
    for frame in range(len(spinors)//frameRate):
        frame_path = f"{frames_dir}/frame_{frame:04d}.png"
        psi = spinors[frame*frameRate]
        xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
        ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   
    

        xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )
        fig, ax = plt.subplots(figsize=(6,6))
        
        match filmType:
            case 'MAG':
                mag = ax.pcolormesh(
                (xMesh),
                (yMesh),
                ( 2*(abs(psi[2])**2-abs(psi[-2])**2) + abs(psi[1])**2 - abs(psi[-1])**2 ),
                vmin=-2, vmax=2 )
                ax.set_aspect('equal')
                fig.colorbar( mag )
            case 'DENS':
                dens = ax.pcolormesh(
                (xMesh),
                (yMesh),
                ( abs(psi[2])**2+abs(psi[-2])**2 + abs(psi[1])**2 + abs(psi[-1])**2 + abs(psi[0])**2  ),
                vmin=-2, vmax=2 )
                ax.set_aspect('equal')
                fig.colorbar( dens )
        
        plt.savefig(frame_path)

        plt.close()

    ani.movieFromFrames( filmName, frames_dir )

def main( recalculate:bool=False ):

    power2 = 7
    # Generate grid object
    points = (2**power2, 2**power2)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)

    # trap = infinitePotential( grid, 2**(power2-1) - 2 * (power2-5), 2**(power2-1) - 2 * (power2-5) )
    circularTrap = circularInfinitePotential( grid, 2**(power2-2) - 2*(power2-5), 100 )
    # Condensate parameters
    params = {
        "c0": 20,
        "c2": -4,
        "c4": 0,
        "p": 0,
        "q": 0,
        "trap": circularTrap,
        "n0": 1,
        # Time params
        "dt": (1) * 1e-2,
        "nt": 100,
        "t": 0,
        'nx':points[0],
        'ny':points[1],
        'dx':grid_spacings[0],
        'dy':grid_spacings[1],
        "frameRate": 1,
    }
    psi = gpe.SpinTwoWavefunction(grid)

    psi.set_wavefunction(0,1,0,0,0)
    psi.add_noise(['plus1','zero','minus1'], 0.0, 1e-4)

    psi.plus2_component[params['trap'] != 0] = 0 
    psi.plus1_component[params['trap'] != 0] = 0 
    psi.zero_component[params['trap'] != 0] = 0
    psi.minus1_component[params['trap'] != 0] = 0
    psi.minus2_component[params['trap'] != 0] = 0

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()


    targetDirectory = "dataSpin2"
    fileName = 'nematicConditionsQ=-1.hdf5'
    filePath = './dataSpin2/' + fileName



    spinors, (mu, lam) = getRelaxation( grid, params, Spinor( psi.plus2_component, psi.plus1_component, psi.zero_component, psi.minus1_component, psi.minus2_component ) )
    params.update({'mu':mu, 'lambda': lam })
    filePath = 'dataSpin2'
    os.makedirs(filePath, exist_ok=True )
    extractStructure( spinors, params, filePath + '/ferroConditions.hdf5', frameRate=params['frameRate'] )

    print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

    # film( spinors, params, 'spin2GroundStates/ferroConditionsQ=-1Mag.mp4', 'frames', frameRate=params['frameRate'], filmType='DENS' )



    # if not os.path.exists( filePath ) or recalculate:
    #     getData( psi, params, fileName, targetDirectory )
    

if __name__ == '__main__':
    main(True)

    # file = h5py.File( './dataSpin2/nematicConditionsQ=-1.hdf5', 'r')
    # waveFunc = file['wavefunction']
    # scalars = hdf5ReadScalars( file )

    # os.makedirs('frames', exist_ok=True)
    # for frame in range(scalars["nt"]//scalars["frameRate"]):
    #     ani.allComponentSpin2Frame(waveFunc, scalars, frame, 'frames')

    # ani.movieFromFrames( 'spin2GroundStates/nematicConditionsQ=-1AllComp.mp4', 'frames' )
    # totalEnergyPlot( waveFunc, scalars )