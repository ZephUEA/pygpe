import pygpe.spinone as gpe
import os
import time
import h5py
import animation as ani
import matplotlib.pyplot as plt
import numpy as np
import pygpe.shared.vortices as vort
import correlation as corr
from scipy.ndimage import label
# np.seterr(all='raise')

def getData( psi:gpe.SpinOneWavefunction, params:dict, fileName:str, dataPath:str ) -> None:
    data = gpe.DataManager(fileName, dataPath, psi, params)
    percentages = int(params['nt']/100)
    for i in range(params["nt"]):

        if i % params["frameRate"] == 0:  # Save wavefunction data and create a frame
            data.save_wavefunction(psi)
        
        if i % percentages ==0 :
            print(f'{i//percentages}% ')
        

        # Evolve wavefunction
        gpe.step_wavefunction(psi, params)

        params["t"] += params["dt"]  # Increment time count

def sgn( x ):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

def sphericalToCartesian( r, theta, phi ):
    x = r * np.sin(theta)*np.cos(phi)
    y = r * np.sin(theta)*np.sin(phi)
    z = r * np.cos(theta) 
    return (x,y,z)

def cartesianToSpherical( x, y, z ):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.acos(z/r)
    phi = np.acos(x/np.sqrt(x**2 + y**2)) * sgn( y )
    return r, theta, phi

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


def counterFlowInitial( grid, relVel, params ):
    psi = gpe.SpinOneWavefunction(grid)
    phase = grid.x_mesh * relVel/2
    psi.set_wavefunction(np.sqrt(params['n0']/2),None,np.sqrt(params['n0']/2))
    psi.apply_phase(phase,'plus')
    psi.apply_phase(-phase,'minus')
    return psi

def phaseGradient( psi, scalars, frame ):
    # Assuming the plus component doesnt lose too much density
    phase = np.angle(psi['psi_plus'][:,:,frame])
    phaseX, phaseY = np.gradient( phase, scalars['dx'],scalars['dy'])
    return np.array( [phaseX, phaseY])

def aField( psi, scalars, frame ):
    # i zeta * nabla zeta
    normalisation = abs(psi['psi_plus'][:,:,frame])**2 + abs(psi['psi_zero'][:,:,frame])**2 + abs(psi['psi_minus'][:,:,frame])**2

    # Assuming the plus component doesnt lose too much density
    phase = np.angle(psi['psi_plus'][:,:,frame])

    plusNorm = psi['psi_plus'][:,:,frame]  /  ( normalisation * phase )
    zeroNorm = psi['psi_zero'][:,:,frame]  /  ( normalisation * phase )
    minusNorm = psi['psi_minus'][:,:,frame] / ( normalisation * phase )

    zetaPlusX, zetaPlusY = np.gradient(plusNorm, scalars['dx'],scalars['dy'])
    zetaZeroX, zetaZeroY = np.gradient(zeroNorm,scalars['dx'],scalars['dy'])
    zetaMinusX, zetaMinusY = np.gradient(minusNorm,scalars['dx'],scalars['dy'])
    aX = np.conj(plusNorm) * zetaPlusX + np.conj(zeroNorm) * zetaZeroX + np.conj(minusNorm) * zetaMinusX 
    aY = np.conj(plusNorm) * zetaPlusY + np.conj(zeroNorm) * zetaZeroY + np.conj(minusNorm) * zetaMinusY

    return np.array([aX.imag, aY.imag])



def spinDistribution(psi, frame, frames_dir, bins=100):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    mz = abs(psi['psi_plus'][:,:,frame])**2-abs(psi['psi_minus'][:,:,frame])**2
    mx = np.conj(psi['psi_plus'][:,:,frame]+psi['psi_minus'][:,:,frame])*psi['psi_zero'][:,:,frame] + np.conj(psi['psi_zero'][:,:,frame])*(psi['psi_plus'][:,:,frame]+psi['psi_minus'][:,:,frame]) 
    my = np.conj(-psi['psi_plus'][:,:,frame]+psi['psi_minus'][:,:,frame])*psi['psi_zero'][:,:,frame] + np.conj(psi['psi_zero'][:,:,frame])*(psi['psi_plus'][:,:,frame]-psi['psi_minus'][:,:,frame]) 
    # For numerics ive moved the factors of sqrt 2 from mx and my into the definition of the spin vector
    s = abs(mz)**2 + (abs(mx)**2)/2 + (abs(my)**2)/2
    plt.hist(s.flatten(),bins)
    plt.xlim(0,1.5)
    plt.ylim(0,600)
    plt.savefig(frame_path)
    plt.close()

def directorFrame(psi, scalars, frame, frames_dir):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(18,6))
    xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )

    directorX = axs[0].pcolormesh(
        (xMesh),
        (yMesh),
        abs( psi["psi_plus"][ :, :, frame ]  - psi['psi_minus'][:,:,frame] )/np.sqrt(2),
        vmin=0, vmax=1.2 )
    fig.colorbar( directorX )

    directorY = axs[1].pcolormesh(
        (xMesh),
        (yMesh),
        abs( psi["psi_plus"][ :, :, frame ] +  psi['psi_minus'][:,:,frame] )/np.sqrt(2) ,
        vmin=0, vmax=1.2 )
    fig.colorbar( directorY )

    directorZ = axs[2].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_zero"][ :, :, frame ] ) ),
        vmin=0, vmax=1.2 )
    fig.colorbar( directorZ )

    plt.savefig(frame_path)

    plt.close()

def relativeArgFrame(psi, frame, frames_dir):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fig,ax = plt.subplots(figsize=(6,6))
    xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )

    argRel = ax.pcolormesh(
        xMesh, yMesh,
        ( np.angle(psi['psi_plus'][:,:,frame]) - np.angle(psi['psi_minus'][:,:,frame]) - np.pi) % (2*np.pi),
    vmin=0, vmax=2*np.pi )
    fig.colorbar(argRel)
    plt.savefig(frame_path)
    plt.close()

def stabPlots(params):
    # Set hbar and mass to 1
    k = np.linspace(0, 10, 1000,dtype='complex128')
    relVel = np.linspace(0, 10, 1000,dtype='complex128')
    
    # Create 2D meshgrid
    K, RelVel = np.meshgrid(k, relVel)
    
    # Compute epsilon using the meshgrid
    epsilon = K**2 / 2

    soundSpeed = np.sqrt(params['c0']/2)
    densityHealingLength = np.sqrt(params['c0']*2)
    
    # Compute your functions over the 2D grid
    stab0 = np.sqrt((epsilon + params['c2']*params['n0'] - (RelVel**2)/8)**2 - (params['c2']*params['n0'])**2)
    
    stabpm = np.sqrt(epsilon**2 + (params['c0'] + params['c2'])*epsilon*params['n0'] + (RelVel*K/2)**2 -
                     np.sqrt(((RelVel*K)**2)*epsilon*(epsilon + params['n0']*(params['c0'] + params['c2'])) + 
                            (epsilon*params['n0']*(params['c0'] - params['c2']))**2))
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot stab0
    im1 = ax1.pcolormesh(K.real/densityHealingLength, RelVel.real/soundSpeed, abs(stab0.imag), shading='auto', cmap='viridis')
    ax1.set_xlabel('k')
    ax1.set_ylabel(r'$V_r/c_s$')
    ax1.set_title('stab0')
    plt.colorbar(im1, ax=ax1)
    
    # Plot stabpm
    im2 = ax2.pcolormesh(K.real/densityHealingLength, RelVel.real/soundSpeed, stabpm.imag, shading='auto', cmap='viridis')
    ax2.set_xlabel('k')
    ax2.set_ylabel(r'$V_r/c_s$')
    ax2.set_title('stabpm')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('instability.png')
    plt.close()


def squaredLatticeMags( psi, scalars, mode ):
    area = scalars['nx']*scalars['ny']
    match mode.upper():
        case 'X':
            x =  np.array(np.conj(psi['psi_plus'][()]+psi['psi_minus'][()])*psi['psi_zero'][()] 
                            + np.conj(psi['psi_zero'][()])*(psi['psi_plus'][()]+psi['psi_minus'][()]))/np.sqrt(2)
            return np.sum( abs(x)**2, axis=(0,1) )/(area*scalars['n0']**2)
        case 'Y':
            y =  1j*np.array(np.conj(-psi['psi_plus'][()]+psi['psi_minus'][()])*psi['psi_zero'][()] 
                            + np.conj(psi['psi_zero'][()])*(psi['psi_plus'][()]-psi['psi_minus'][()]))/np.sqrt(2)
            return np.sum( abs(y)**2, axis=(0,1) )/(area*scalars['n0']**2)
        case 'Z':
            z =  np.array(abs(psi['psi_plus'][()])**2 - abs(psi['psi_minus'][()])**2)
            return np.sum( abs(z)**2, axis=(0,1) )/(area*scalars['n0']**2)
        case 'T':
            return squaredLatticeMags(psi, scalars, 'X' ) + squaredLatticeMags(psi, scalars, 'Y') + squaredLatticeMags(psi, scalars, 'Z')
        case _:
            raise ValueError('Not a valid mode')

def orderParameterPlots( psi, scalars ):
    x = squaredLatticeMags(psi, scalars, 'X')
    y = squaredLatticeMags(psi, scalars, 'Y')
    z = squaredLatticeMags(psi, scalars, 'Z')
    t = x + y + z
    xaxis = np.linspace(0, scalars['dt']*scalars['nt'], scalars['nt']//scalars['frameRate'])
    plt.plot( xaxis, x)
    plt.plot( xaxis, y)
    plt.plot( xaxis, z)
    plt.plot( xaxis, t)
    plt.legend([r'$F_x$',r'$F_y$',r'$F_z$',r'$F_t$'])
    plt.show()

def smoothing( arr, values ):
    for i in range( arr.shape[-1] -1 ):
        arr[:,:,i] = np.sum( arr[:,:,i:min( i+values, arr.shape[-1]-1)],axis=-1 ) / min( values, arr.shape[-1]-i)
    return arr


def temperoSpatialAverage(psi, scalars, values=None, mode='MAG'):
    
    area = scalars['nx']*scalars['ny']
    match mode.upper():
        case 'MAG':
            x =  np.array(np.conj(psi['psi_plus'][()]+psi['psi_minus'][()])*psi['psi_zero'][()] 
                            + np.conj(psi['psi_zero'][()])*(psi['psi_plus'][()]+psi['psi_minus'][()]))/np.sqrt(2)
            y =  1j*np.array(np.conj(-psi['psi_plus'][()]+psi['psi_minus'][()])*psi['psi_zero'][()] 
                            + np.conj(psi['psi_zero'][()])*(psi['psi_plus'][()]-psi['psi_minus'][()]))/np.sqrt(2)
            z =  np.array(abs(psi['psi_plus'][()])**2 - abs(psi['psi_minus'][()])**2)
        case 'DIR':
            z = np.array(psi["psi_zero"][()])
            angle = np.angle(z)
            z = abs(z)
            x = np.array(psi["psi_minus"][()]  - psi['psi_plus'][()]/np.sqrt(2)) * np.exp(1j*angle)
            y = np.array(psi["psi_plus"][()]  + psi['psi_minus'][()]/(np.sqrt(2)*1j)) * np.exp(1j*angle)

    # xCum = np.flip(np.cumsum(np.flip(x,axis=-1),axis=-1)/np.arange(1,x.shape[-1]+1), axis=-1)
    # yCum = np.flip(np.cumsum(np.flip(y,axis=-1),axis=-1)/np.arange(1,x.shape[-1]+1),axis=-1)
    # zCum = np.flip(np.cumsum(np.flip(z,axis=-1),axis=-1)/np.arange(1,x.shape[-1]+1),axis=-1)
    if values is None:
        values = z.shape[-1]
    xCum = smoothing( x, values)
    yCum = smoothing( y, values)
    zCum = smoothing( z, values)

    orderParam = ( np.sum(xCum.real,axis=(0,1))/area, np.sum(yCum.real,axis=(0,1))/area, np.sum(zCum,axis=(0,1))/area )
    orderParamSMS = ( np.sum(abs(xCum)**2,axis=(0,1)) + np.sum(abs(yCum)**2,axis=(0,1)) + np.sum(abs(zCum)**2,axis=(0,1)) )/area
    return orderParam, orderParamSMS


def correlationFunction( psi, scalars, inverse=False ):
    nx = scalars["nx"]
    ny = scalars["ny"]
    plusComp = np.array( psi['psi_plus'][()] )
    zeroComp = np.array( psi['psi_zero'][()] )
    minusComp = np.array( psi['psi_minus'][()] )
    plusFFT = np.fft.fft2( plusComp, axes=(0,1) )
    zeroFFT = np.fft.fft2( zeroComp, axes=(0,1) )
    minusFFT = np.fft.fft2( minusComp, axes=(0,1) )
    dotProduct = abs( plusFFT ) ** 2 + abs( zeroFFT )**2 + abs( minusFFT )**2
    if inverse:
        return dotProduct
    autoCorrelation = np.fft.ifft2(dotProduct, axes=(0,1) ).real
    return autoCorrelation / ( nx * ny )

def plotCorrelation( psi, scalars, frames_dir, inverse=False ):
    if inverse:
        dotProducts = correlationFunction( psi, scalars, True )
        xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
        ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

        xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )

        for frame in range( dotProducts.shape[-1] ):
            frame_path = f"{frames_dir}/frame_{frame:04d}.png"
            fig,ax = plt.subplots()
            chart = ax.pcolormesh( xMesh, yMesh, dotProducts[:,:,frame] )
            fig.colorbar( chart )
            plt.savefig(frame_path)
            plt.close()
    else:
        radii = np.linspace(0, scalars['dx']*scalars['nx']//2, scalars['nx'] // 2 )
        circles = corr.createCircles( ( scalars['nx'], scalars['ny'] ), stepsize=1, threshold=128 )
        correlations = ( np.sum( correlationFunction( psi, scalars )[:,:,None,:] * circles[:,:,:,None], axis=(0,1) ) )
        for frame in range( correlations.shape[-1] ):
            frame_path = f"{frames_dir}/frame_{frame:04d}.png"
            fig,ax = plt.subplots()
            plt.plot( radii, correlations[:,frame] )
            plt.xlim(0,scalars['dx']*scalars['nx']//2)
            plt.ylim(-0.5,1)
            plt.savefig(frame_path)
            plt.close()

def skyrmionNumber( psi, scalars, frame, frames_dir ):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(18,12))
    xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )

    spinZ = np.array( abs( psi['psi_plus'][:,:,frame] )**2 - abs( psi['psi_minus'][:,:,frame] )**2 )
    spinX = np.array(np.conj(psi['psi_plus'][:,:,frame]+psi['psi_minus'][:,:,frame])*psi['psi_zero'][:,:,frame] 
                            + np.conj(psi['psi_zero'][:,:,frame])*(psi['psi_plus'][:,:,frame]+psi['psi_minus'][:,:,frame]))/np.sqrt(2)
    spinY = 1j*np.array(np.conj(-psi['psi_plus'][:,:,frame]+psi['psi_minus'][:,:,frame])*psi['psi_zero'][:,:,frame] 
                            + np.conj(psi['psi_zero'][:,:,frame])*(psi['psi_plus'][:,:,frame]-psi['psi_minus'][:,:,frame]))/np.sqrt(2)
    
    spinVector = np.array([spinX.real, spinY.real, spinZ.real])

    xx,xy = np.gradient(spinX.real)
    yx,yy = np.gradient(spinY.real)
    zx,zy = np.gradient(spinZ.real)

    partialX = np.array([xx,yx,zx])
    partialY = np.array([xy,yy,zy])
    crossPartials = np.cross(partialX,partialY,axisa=0,axisb=0,axisc=0)
    skyrme = spinVector[0]*crossPartials[0] + spinVector[1]*crossPartials[1] + spinVector[2]*crossPartials[2]
    divergence = xx + yy
    curlZ = yx - xy
    curlX = zy
    curlY = -zx
    skyrmePlot = axs[0][0].pcolormesh(
        (xMesh),
        (yMesh),
        skyrme,
         vmin=-0.1, vmax=0.1 )
    fig.colorbar( skyrmePlot )

    divPlot = axs[0][1].pcolormesh(
        (xMesh),
        (yMesh),
        divergence,
        vmin=-1, vmax=1)
    fig.colorbar( divPlot )
    curlZPlot = axs[1][0].pcolormesh(
        (xMesh),
        (yMesh),
        curlZ,
        vmin=-1, vmax=1)
    fig.colorbar( curlZPlot )
    curlXPlot = axs[1][1].pcolormesh(
        (xMesh),
        (yMesh),
        curlX,
        vmin=-1, vmax=1)
    fig.colorbar( curlXPlot )
    curlYPlot = axs[1][2].pcolormesh(
        (xMesh),
        (yMesh),
        curlY,
        vmin=-1, vmax=1)
    fig.colorbar( curlYPlot )
    plt.savefig(frame_path)

    plt.close()

def densityFrame(psi, scalars, frame, frames_dir ):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fig,ax = plt.subplots(figsize=(6,6))
    xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )

    density = np.sqrt( abs(psi['psi_plus'][:,:,frame])**2+abs(psi['psi_zero'][:,:,frame])**2+abs(psi['psi_minus'][:,:,frame])**2 )
    densityPlot = ax.pcolormesh(
        (xMesh),
        (yMesh),
        density,
        vmin=0.0, vmax=2 )
    fig.colorbar( densityPlot )

    plt.savefig(frame_path)

    plt.close()


def removePhaseDiscontinuity( phase ):
    xlen, ylen = phase.shape
    xs = np.array([i for i in range(xlen)]) / xlen
    ys = np.array([i for i in range(ylen)]) / ylen
    xPhaseDifference = ( phase[0,:] - phase[-1,:] ) % (2 * np.pi) #TODO fix when we would have small negative numbers, which become large numbers after mod
    yPhaseDifference = ( phase[:,0] - phase[:,-1] ) % (2 * np.pi)
    newPhase =  phase  + yPhaseDifference[:,None] * ys[None,:]
    return newPhase


def skyrmionInitial( grid, coord1, coord2, radius ):
    
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
    phase2 = vort._calculate_vortex_contribution(grid, coord2[0],coord2[1],1) 
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
    beta1 = np.pi * ( (np.sign(r1-radius/2)+1)/2 + (np.sign(r1-radius)+1)/2 )/2

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
    trap[abs(r)>radius] = 1e10
    return trap


def radialVelocityFrame( psi, scalars, frame, frames_dir ):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"

    psiPlus = psi['psi_plus'][:,:,frame]
    psiZero = psi['psi_zero'][:,:,frame]
    psiMinus = psi['psi_minus'][:,:,frame]
    plusX, plusY = np.gradient( psiPlus, scalars['dx'], scalars['dy'], axis=(0,1))
    zeroX, zeroY = np.gradient( psiZero, scalars['dx'], scalars['dy'], axis=(0,1))
    minusX, minusY = np.gradient( psiMinus, scalars['dx'], scalars['dy'], axis=(0,1))
    velocityX = ( np.conj(psiPlus)*plusX + np.conj(psiZero)*zeroX +  np.conj(psiMinus)*minusX ).imag
    velocityY = ( np.conj(psiPlus)*plusY + np.conj(psiZero)*zeroY +  np.conj(psiMinus)*minusY ).imag
    speed = np.sqrt(abs(velocityX)**2 + abs(velocityY)**2)
    cosBeta = (abs(psiPlus)**2 - abs(psiMinus)**2)/(abs(psiPlus)**2 + abs(psiZero)**2 + abs(psiMinus)**2)
    beta = np.arccos( cosBeta )
    xs = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = np.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   
    

    xMesh, yMesh = np.meshgrid( xs, ys, indexing='ij' )
    r = np.sqrt( abs(xMesh)**2 + abs(yMesh)**2 )

    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,6))

    speedPlot = axs[0].pcolormesh(
        (xMesh),
        (yMesh),
        ( speed ),
        vmin=0 )
    axs[0].set_aspect('equal')
    axs[0].set_title('Numerics')
    fig.colorbar( speedPlot )


    theoryPlot = axs[1].pcolormesh(
        (xMesh),
        (yMesh),
        np.nan_to_num( (2 * np.sin(beta/2)**2)/r, posinf=0 ),
        vmin=0 )
    axs[1].set_aspect('equal')
    axs[1].set_title('Theory')
    fig.colorbar( theoryPlot )

    plt.savefig(frame_path)

    plt.close()


def extractRadialProfile( psi, scalars, frame ):
    psiPlus = psi['psi_plus'][:,:,frame]
    psiZero = psi['psi_zero'][:,:,frame]
    psiMinus = psi['psi_minus'][:,:,frame]
    halfXPoint = psiPlus.shape[0]//2

    radius = np.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']

    cosBeta = (abs(psiPlus)**2 - abs(psiMinus)**2)/(abs(psiPlus)**2 + abs(psiZero)**2 + abs(psiMinus)**2)
    beta = np.arccos( cosBeta )
    radialBeta = beta[:, beta.shape[1]//2]
    m = corr.bestFitCurveError( lambda x, m: m*x , radius[halfXPoint:halfXPoint+15],radialBeta[halfXPoint:halfXPoint+15] )
    p,a = corr.bestFitCurveError(lambda x, p, A: np.pi - A*np.exp(-p*x)/np.sqrt(x), radius[halfXPoint+10:-20], radialBeta[halfXPoint+10:-20] )
    r = corr.bestFitCurveError( lambda x, r: np.pi * np.tanh(x/r) ,radius[halfXPoint:-20], radialBeta[halfXPoint:-20])
    plt.plot( radius, radialBeta )
    plt.plot( radius[halfXPoint:halfXPoint+15], radius[halfXPoint:halfXPoint+15]*m[0][0])
    plt.plot( radius[halfXPoint+10:-10], np.pi - a[0] * ( np.exp(-p[0] * radius[halfXPoint+10:-10] ) / np.sqrt(radius[halfXPoint+10:-10] ) ) )
    # plt.plot( radius[halfXPoint:-20], np.pi * np.tanh(radius[halfXPoint:-20]/r[0][0]) )
    plt.hlines( np.pi, radius[0],radius[-1], colors=['k'])
    plt.legend(['Data',f'Linear Core {m[0][0]:.2f}r', fr'$\pi-{a[0]:.2f}exp(-{p[0]:.2f}r)/\sqrt{{r}}$', f'$\pi$'])
    plt.show()


def main( recalculate:bool=False ):

    power2 = 8
    # Generate grid object
    points = (2**power2, 2**power2)
    grid_spacings = (0.5,0.5)
    grid = gpe.Grid(points, grid_spacings)

    trap = infinitePotential( grid, 2**(power2-1) - 2 * (power2-5), 2**(power2-1) - 2 * (power2-5) )
    circularTrap = circularInfinitePotential( grid, 2**(power2-2) - 2*(power2-5))
    
    # Condensate parameters
    params = {
        "c0": 108, # 1 is about right for lithium while 108 is for rubidium
        "c2": -0.5,
        "p": 0,
        "q": 0,
        "trap": circularTrap,
        "n0": 1,
        'qSpace': 0,
        # Time params
        "dt": (2-1e-5j) * 1e-2,
        "nt": 100_000,
        "t": 0,
        "frameRate": 1000,
    }

    # Generate wavefunction object, set initial state and add noise
    # I think the speed of sound is n*c_0 in our natural units.

    # psi = counterFlowInitial( grid, relVelocity, params )
    # psi = vortexPairInitial( grid,(10,0), (-10,0) )

    # psi = skyrmionInitial( grid, (20,0), (-20,0), 30 )
    psi = singleSkyrmion( grid, (0,0), 10 )

    psi.add_noise("all", 0.0, 1e-4)
    psi.plus_component[params['trap'] != 0] = 0 
    psi.zero_component[params['trap'] != 0] = 0
    psi.minus_component[params['trap'] != 0] = 0

    psi._update_atom_numbers()

    psi.fft()  # Ensures k-space wavefunction components are up-to-date before evolution
    start_time = time.time()

    # psi, params = initialCondtiions(1,2,1)

    targetDirectory = "dataSpin1"
    fileName = 'singleSkyrmionStepStructureImag.hdf5' 
    filePath = './dataSpin1/' + fileName

    if not os.path.exists( filePath ) or recalculate:
        getData( psi, params, fileName, targetDirectory )
        print(f'Evolution of {params["nt"]} steps took {time.time() - start_time}!')

if __name__ == '__main__':
    main(False)

    file = h5py.File( './dataSpin1/singleSkyrmionStepStructureImag.hdf5', 'r')
    waveFunc = file['wavefunction']
    scalars = hdf5ReadScalars( file )
    
    # magnetisationPlots(waveFunc, scalars, 500, 'frames' )
    
    os.makedirs('frames', exist_ok=True)
    for frame in range(scalars["nt"]//scalars["frameRate"]):
        # ani.takeFrame( waveFunc, scalars, 'frames', frame, 'MAG',[] )
        # densityFrame( waveFunc, scalars, frame, 'frames')
        # skyrmionNumber( waveFunc, scalars, frame, 'frames' )
        # ani.magnetisationQuiverFrame( waveFunc, scalars, frame, 'frames' )
        # ani.superfluidVelocitiesFrame(waveFunc, scalars, frame, 'frames')
        # magnetisationModulationFrame( waveFunc, scalars, frame, 'frames' )
        ani.allComponentFrame(waveFunc, scalars, frame, 'frames')
        # ani.allArgsFrame( waveFunc, scalars, frame, 'frames')
        # ani.allArgsChemPotFrame( waveFunc, scalars, frame, 'frames' )
        # radialVelocityFrame( waveFunc, scalars, frame, 'frames')
        # relativeArgFrame( waveFunc, frame, 'frames')
        # spinDistribution( waveFunc, frame, 'frames', bins=1000)
        pass
    
    # plotCorrelation( waveFunc, scalars, 'frames' )
    ani.movieFromFrames( 'initialSkyrme/singleSkyrmionStepStructureImagAllComp.mp4', 'frames' )
    # extractRadialProfile( waveFunc, scalars, -10 )