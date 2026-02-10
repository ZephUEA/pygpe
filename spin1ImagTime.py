import pygpe.spinone as gpe
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pygpe.shared.vortices as vort
from pygpe.spinone.relaxation import SpinorBECGroundState2D, Spinor
import animation as ani
import h5py
import correlation as corr

def sgn( x ):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


# np.seterr(all='raise')
def getData( grid:gpe.Grid, params:dict, psi ) -> None:
    system = SpinorBECGroundState2D( grid, params, psi )
    percentages = int(params['nt']/100)
    initials = (np.sum(abs(system.waveFunctions[-1][1])**2 +abs(system.waveFunctions[-1][0])**2+ abs(system.waveFunctions[-1][-1])**2 ),
                np.sum(abs(system.waveFunctions[-1][1])**2 - abs(system.waveFunctions[-1][-1])**2 )  )
    
    for i in range(params["nt"]):
        if i % percentages == 0 :
            print(f'{i//percentages}% ')
        #     print( f'N = {np.sum(abs(system.waveFunctions[-1][1])**2 +abs(system.waveFunctions[-1][0])**2+ abs(system.waveFunctions[-1][-1])**2 )}')
        #     print( f'MagZ = {np.sum(abs(system.waveFunctions[-1][1])**2 - abs(system.waveFunctions[-1][-1])**2 )}')
        

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


def skyrmionInitial( grid, coord1, coord2, radius, winding=1 ):
    
    r1 = np.sqrt( (grid.x_mesh-coord1[0])**2 + (grid.y_mesh-coord1[1])**2 )
    r2 = np.sqrt( (grid.x_mesh-coord2[0])**2 + (grid.y_mesh-coord2[1])**2 )
    beta1 = np.pi * (np.tanh( r1/radius ))
    beta2 = np.pi * (np.tanh( r2/radius ))

    plusComp =  ( ((np.cos(beta1/2))**2 + (np.cos(beta2/2))**2) ) / 2
    zeroComp = ( np.sin(beta1) /np.sqrt(2) + np.sin(beta2) / np.sqrt(2) ) /2
    minusComp = ((np.sin(beta1/2))**2  * (np.sin(beta2/2))**2 )

    norm = np.sqrt( abs(plusComp)**2 + abs(zeroComp) **2 + abs(minusComp)**2 )

    plusComp /= norm
    zeroComp /= norm
    minusComp /= norm

    psi = gpe.SpinOneWavefunction(grid)
    psi.set_wavefunction(plusComp,zeroComp,minusComp)
    phase1 = vort._calculate_vortex_contribution(grid, coord1[0],coord1[1],1)
    phase2 = vort._calculate_vortex_contribution(grid, coord2[0],coord2[1], sgn(winding) ) 
    phaseTotal = phase1 + phase2
    # phaseTotal = removePhaseDiscontinuity( phaseTotal )
    psi.apply_phase( phaseTotal,['zero','minus'] )
    psi.apply_phase( phaseTotal, 'minus' )
    return psi

def vortexPairInitial( grid, coord1, coord2 ):
    psi = gpe.SpinOneWavefunction( grid )
    minusComp = np.ones( grid.x_mesh.shape )
    zeroComp = np.zeros( grid.x_mesh.shape )
    minusComp[ abs(grid.x_mesh-coord1[0])**2 + abs(grid.y_mesh-coord1[1]**2) <=1 ] = 0 
    zeroComp[ abs(grid.x_mesh-coord1[0]) + abs(grid.y_mesh-coord1[1]**2) <=1 ] = 1
    minusComp[ abs(grid.x_mesh-coord2[0]) + abs(grid.y_mesh-coord2[1]**2) <=1 ] = 0 
    zeroComp[ abs(grid.x_mesh-coord2[0]) + abs(grid.y_mesh-coord2[1]**2) <=1 ] = 1

    psi.set_wavefunction(minus_component=minusComp, zero_component=zeroComp)
    phase1 = vort._calculate_vortex_contribution(grid, coord1[0],coord1[1],1)
    phase2 = vort._calculate_vortex_contribution(grid, coord2[0],coord2[1],-1)
    phaseTotal = phase1 + phase2
    # phaseTotal = removePhaseDiscontinuity( phaseTotal )
    psi.apply_phase( phaseTotal, 'minus')
    return psi


def singleSkyrmion(grid, coord1, radius):
    r1 = np.sqrt( (grid.x_mesh-coord1[0])**2 + (grid.y_mesh-coord1[1])**2 )
    beta1 = np.pi * (np.tanh( r1/radius ))
    # beta1 = np.pi * ( (np.sign(r1-radius/2)+1)/2 + (np.sign(r1-radius)+1)/2 )/2

    plusComp =  (np.cos(beta1/2))**2  
    zeroComp =  np.sin(beta1) /np.sqrt(2) 
    minusComp = (np.sin(beta1/2))**2 

    psi = gpe.SpinOneWavefunction(grid)
    psi.set_wavefunction(plusComp,zeroComp,minusComp)
    phase1 = vort._calculate_vortex_contribution(grid, coord1[0],coord1[1],1)
    phaseTotal = phase1 
    # phaseTotal = removePhaseDiscontinuity( phaseTotal )
    psi.apply_phase( phaseTotal,['zero','minus'] )
    psi.apply_phase( phaseTotal, 'minus' )
    return psi

def harmonicPotential( grid, trapLength ):
    r = np.sqrt( abs(grid.x_mesh)**2 + abs(grid.y_mesh)**2 )
    return r**2 /(2*trapLength**2)


def infinitePotential( grid, xWidth, yWidth ):
    x = grid.x_mesh
    y = grid.y_mesh

    trap = np.zeros(x.shape)
    trap[abs(x)>xWidth/2] = 1e10
    trap[abs(y)>yWidth/2] = 1e10

    return trap

def circularInfinitePotential( grid, radius ):
    x = grid.x_mesh
    y = grid.y_mesh
    r = np.sqrt(abs(x)**2 + abs(y)**2)

    trap = np.zeros(r.shape)
    trap[abs(r)>radius] = 100
    return trap


def totalEnergyPlot( psi, scalars ):
    energies = []
    for frame in range( scalars['nt'] ):
        psiPlus = psi[frame][1]
        psiZero = psi[frame][0]
        psiMinus = psi[frame][-1]

        gradPlusX, gradPlusY = np.gradient( psiPlus )
        gradZeroX, gradZeroY = np.gradient( psiZero )
        gradMinusX, gradMinusY = np.gradient( psiMinus )

        gradEnergy = np.sum( np.conj( gradPlusX ) * gradPlusX + np.conj(gradPlusY) * gradPlusY + 
                    np.conj( gradZeroX ) * gradZeroX + np.conj(gradZeroY) * gradZeroY + 
                    np.conj( gradMinusX ) * gradMinusX + np.conj(gradMinusY) * gradMinusY )
        
        densEnergy = np.sum( ( abs(psiPlus)**2 + abs(psiZero)**2 + abs(psiMinus)**2 ) ** 2 )

        magXEnergy = ( ( np.conj(psiPlus) + np.conj(psiMinus) ) * psiZero + np.conj(psiZero)*(psiPlus + psiMinus) )/ np.sqrt(2)
        magYenergy = 1j * ( ( -np.conj(psiPlus) + np.conj(psiMinus) ) * psiZero + np.conj(psiZero)*(psiPlus - psiMinus)) / np.sqrt(2)
        magZEnergy = abs(psiPlus)**2 - abs(psiMinus)**2

        magEnergy = np.sum( abs(magXEnergy) ** 2 + abs(magYenergy) ** 2 + abs(magZEnergy) ** 2 )

        energies.append( gradEnergy + scalars['c0'] * densEnergy + scalars['c2'] * magEnergy )

    ts = np.linspace( 0, scalars['dt']*scalars['nt'], scalars['nt'] )
    plt.plot( abs(ts), np.array(energies).real )
    plt.show()

def magFilm( spinors, scalars, filmName, frames_dir, frameRate=1 ):
    os.makedirs(frames_dir, exist_ok=True)
    for frame in range(len(spinors)//frameRate):
        frame_path = f"{frames_dir}/frame_{frame:04d}.png"
        psi = spinors[frame*frameRate]
        psiPlus = psi[1]
        psiMinus = psi[-1]
        xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
        ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   
    

        xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )
        fig, ax = plt.subplots(figsize=(6,6))
        

        mag = ax.pcolormesh(
        (xMesh),
        (yMesh),
        ( abs(psiPlus)**2 - abs(psiMinus)**2 ),
        vmin=-1, vmax=1 )
        ax.set_aspect('equal')
        fig.colorbar( mag )
        
        plt.savefig(frame_path)

        plt.close()

    ani.movieFromFrames( filmName, frames_dir )

def radialFilm( spinors, scalars, filmName, frames_dir, frameRate=1 ):
    os.makedirs(frames_dir, exist_ok=True)

    for frame in range(len(spinors)//frameRate):
        frame_path = f"{frames_dir}/frame_{frame:04d}.png"
        psi = spinors[frame*frameRate]

        radius = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']

        cosBeta = (abs(psi[1])**2 - abs(psi[-1])**2)/(abs(psi[1])**2 + abs(psi[0])**2 + abs(psi[-1])**2)
        beta = np.arccos( cosBeta )
        radialBeta = beta[:, beta.shape[1]//2]
        plt.plot( radius, radialBeta )
        plt.hlines( np.pi, radius[0],radius[-1], colors=['k'])
        
        plt.savefig(frame_path)

        plt.close()

    ani.movieFromFrames( filmName, frames_dir )
    
def extractStructure(spinors, scalars, filePath, frameRate=1):
    cut = slice(0,None,frameRate)
    saveSpinors = spinors[cut]
    spinorPlus = np.array(list(map( lambda x: x[1], saveSpinors ) ) )
    spinorZero = np.array(list(map( lambda x: x[0], saveSpinors ) ) )
    spinorMinus =np.array( list(map( lambda x: x[-1], saveSpinors ) ) )

    with h5py.File(filePath, 'w') as file:
        # Create the groups first
        params_group = file.create_group("parameters")
        wavefunction_group = file.create_group("wavefunction")
        
        # Add scalar parameters
        for key in scalars:
            params_group.create_dataset(key, data=scalars[key])
        
        # Create and populate wavefunction datasets
        wavefunction_group.create_dataset(
            'psi_plus',
            data=spinorPlus,
            dtype="complex128",
        )
        wavefunction_group.create_dataset(
            'psi_zero',
            data=spinorZero,
            dtype="complex128",
        )
        wavefunction_group.create_dataset(
            'psi_minus',
            data=spinorMinus,
            dtype="complex128",
        )

    
    
def plotStructureFromFile( filename ):
    file = h5py.File( filename, 'r')
    psi = file['wavefunction']
    scalars = hdf5ReadScalars( file )

    psiPlus = psi['psi_plus'][:,:,-1]
    psiZero = psi['psi_zero'][:,:,-1]
    psiMinus = psi['psi_minus'][:,:,-1]
    halfXPoint = psiPlus.shape[0]//2

    radius = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']

    cosBeta = (abs(psiPlus)**2 - abs(psiMinus)**2)/(abs(psiPlus)**2 + abs(psiZero)**2 + abs(psiMinus)**2)
    beta = np.arccos( cosBeta )
    radialBeta = beta[:, beta.shape[1]//2]
    p = np.sqrt(abs(scalars['lambda']))
    m = corr.bestFitCurveError( lambda x, m: m*x , radius[halfXPoint:halfXPoint+10],radialBeta[halfXPoint:halfXPoint+10] )
    a = corr.bestFitCurveError(lambda x, A: np.pi - A*np.exp(-p*x)/np.sqrt(x), radius[halfXPoint+10:], radialBeta[halfXPoint+10:]) 
    # r = corr.bestFitCurveError( lambda x, r: np.pi * np.tanh(x/r) ,radius[halfXPoint:-20], radialBeta[halfXPoint:-20])
    plt.plot( radius, radialBeta )
    plt.plot( radius[halfXPoint:halfXPoint+10], radius[halfXPoint:halfXPoint+10]*m[0][0])
    plt.plot( radius[halfXPoint+10:], np.pi - a[0][0] * ( np.exp(-p * radius[halfXPoint+10:] ) / np.sqrt(radius[halfXPoint+10:] ) ) )
    # plt.plot( radius[halfXPoint:-20], np.pi * np.tanh(radius[halfXPoint:-20]/r[0][0]) )
    plt.hlines( np.pi, radius[0],radius[-1], colors=['k'])
    plt.legend(['Data',f'Linear Core {m[0][0]:.2f}r', fr'$\pi-{a[0][0]:.2f}exp(-{p:.2f}r)/\sqrt{{r}}$', fr'$\pi$'])
    plt.show()

def main():

    power2 = 7
    # Generate grid object
    points = (2**power2, 2**power2)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)

    # trap = infinitePotential( grid, 2**(power2-1) - 2 * (power2-5), 2**(power2-1) - 2 * (power2-5) )
    circularTrap = circularInfinitePotential( grid, 2**(power2-2) - 2*(power2-5) )
    
    # Condensate parameters
    params = {
        "c0": 20, # 1 is about right for lithium while 108 is for rubidium
        "c2": -0.5,
        "trap": circularTrap,
        # Time params
        "dt": (1) * 1e-2,
        "nt": 40_000,
        "t":0,
        'n0':1,
        'nx':points[0],
        'ny':points[1],
        'dx':grid_spacings[0],
        'dy':grid_spacings[1]
    }

    psi = skyrmionInitial( grid, (10,0), (-10,0), 5, winding=-1 )
    # psi = singleSkyrmion( grid, (0,0), 5 )

    psi.add_noise("all", 0.0, 1e-4)
    psi.plus_component[params['trap'] != 0] = 0 
    psi.zero_component[params['trap'] != 0] = 0
    psi.minus_component[params['trap'] != 0] = 0

    psi._update_atom_numbers()

    start_time = time.time()

    spinors, (mu, lam) = getData( grid, params, Spinor( psi.plus_component, psi.zero_component, psi.minus_component ) )
    params.update({'mu':mu, 'lambda': lam })

    print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

    frameRate = 100

    magFilm( spinors, params, 'initialSkyrme/dualSkyrmionRelaxImagAntiMag.mp4', 'frames', frameRate=frameRate )

        # radialFilm( spinors, params, 'initialSkyrme/imagTanhRad.mp4', 'frames')

    filePath = 'dataInitialSkyrme'
    os.makedirs(filePath, exist_ok=True )
    extractStructure( spinors, params, filePath + '/pairVortexStructureAnti.hdf5', frameRate=frameRate )


    

if __name__ == '__main__':
    main()
    # plotStructureFromFile('dataInitialSkyrme/vortexStructure.hdf5')

