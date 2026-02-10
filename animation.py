try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
from pathlib import Path
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Applications/ffmpeg"
import matplotlib.pyplot as plt
from pygpe.shared.utils import handle_array
try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp
import correlation as corr

def prettyString( params:dict, show:list[str] ) -> str:
    string = f''
    for key,value in params.items():
        if key not in show:
            continue
        string += f'{key} = {value:.3f}, '
    return string[:-2]

def averageFile( fileName:str ) -> float:
    with open( fileName, 'r' ) as file:
        total = 0
        for line in file.readlines():
            total += float( line.strip() )
    return total / len( file.readlines() )

def chemicalArgument( phase:str, scalars:dict ) -> cp.ndarray:
    '''Returns the chemical potential at each point in time'''
    if scalars['dt'].imag == 0:
        timeArray = cp.array( [ n for n in range( scalars['nt'] // scalars['frameRate'] ) ] ) * scalars['dt'] * scalars['frameRate']
    else:
        return cp.zeros((scalars['nt']//scalars['frameRate']))
    match phase.upper():
        case 'FM':
            return timeArray * ( scalars['q'] - abs( scalars['p'] ) + ( scalars['c0'] + scalars['c2'] ) * scalars['n0'] )
        case 'POLAR':
            return timeArray * scalars['c0'] * scalars['n0']
        case 'AFM':
            return timeArray * ( scalars['q'] + scalars['c0'] * scalars['n0'])
        case 'BA':
            return timeArray * ( ( scalars['c0'] + scalars['c2'] ) * scalars['n0'] + scalars['q'] / 2 - ( scalars['p'] ** 2 /( 2 * scalars['q'] ) ) )
        case _:
            raise ValueError(f"{phase} is not a supported ground state")
        


def takeFrame( psi, scalars:dict, frames_dir:str, frame:int, chartType:str, titleElements:list[str], groundState:str='FM', **kwargs ) -> None:

    frame_path = f"{frames_dir}/frame_{frame:04d}.png"

    fig, ax = plt.subplots( figsize=(12, 12) )

    try:
        xs = cp.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
        ys = cp.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

        xMesh, yMesh = cp.meshgrid( xs, ys, indexing='ij' )

    except:
        pass
    
    chemArg = chemicalArgument( groundState, scalars )

    match chartType.upper():
        case 'MAG':
            magnetisation = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( abs( psi["psi_plus"][ :, :, frame ] ) ** 2 - abs( psi["psi_minus"][ :, :, frame ] ) ** 2 ), 
                vmin=-1, vmax=1 )
            fig.colorbar( magnetisation )

        case 'VORT_PLUS':
            plusVorticity = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( psi['psi_plus'][:,:,frame] ) )
            fig.colorbar( plusVorticity )

        case 'VORT_ZERO':
            zeroVorticity = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( psi['psi_zero'][:,:,frame] ))
            fig.colorbar( zeroVorticity )

        case 'VORT_MINUS':
            minusVorticity = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( psi['psi_minus'][:,:,frame] ) )
            fig.colorbar( minusVorticity )

        case 'DENS_PLUS':
            densityPlus = ax.pcolormesh(
                handle_array(xMesh),
                handle_array(yMesh),
                abs(handle_array( psi["psi_plus"][ :, :, frame ] ) ) ** 2,
                vmin=0, vmax=1 )
            fig.colorbar( densityPlus )

        case 'DENS_ZERO':
            densityZero = ax.pcolormesh(
                handle_array(xMesh),
                handle_array(yMesh),
                abs(handle_array( psi["psi_zero"][ :, :, frame ] ) ) ** 2,
                vmin=0, vmax=1 )
            fig.colorbar( densityZero )
            
        case 'DENS_MINUS':
            densityMinus = ax.pcolormesh(
                handle_array(xMesh),
                handle_array(yMesh),
                abs(handle_array( psi["psi_minus"][ :, :, frame ] ) ) ** 2,
                vmin=0, vmax=1 )
            fig.colorbar( densityMinus )
        
        case 'DENS':
            density = ax.pcolormesh(
                handle_array(xMesh),
                handle_array(yMesh),
                cp.sqrt( abs( handle_array(  psi["psi_plus"][ :, :, frame ] ) ) ** 2+ abs(psi["psi_zero"][ :, :, frame ] ) ** 2+ abs( psi["psi_minus"][ :, :, frame ] ) ** 2),
                vmin=0, vmax=2 )
            fig.colorbar( density )

        case 'ARG_PLUS':
            argPlus = ax.pcolormesh(
                handle_array(xMesh),
                handle_array(yMesh),
                handle_array( cp.angle( psi["psi_plus"][ :, :, frame ] * cp.exp( 1j * chemArg )[ None, None, frame ]  ) ),
                cmap="jet", vmin=-cp.pi, vmax=cp.pi )
            fig.colorbar( argPlus )

        case 'ARG_ZERO':
            argZero = ax.pcolormesh(
                handle_array(xMesh),
                handle_array(yMesh),
                handle_array(cp.angle( psi["psi_zero"][ :, :, frame ] * cp.exp( 1j * chemArg )[ None, None, frame ] ) ),
                cmap="jet", vmin=-cp.pi, vmax=cp.pi )
            fig.colorbar( argZero )

        case 'ARG_MINUS':
            argMinus = ax.pcolormesh(
                handle_array(xMesh),
                handle_array(yMesh),
                handle_array(cp.angle( psi["psi_minus"][ :, :, frame ] * cp.exp( 1j * chemArg )[ None, None, frame ] ) ),
                cmap="jet", vmin=-cp.pi, vmax=cp.pi )
            fig.colorbar( argMinus )

        case 'MAG_TRANS_DENS':
            transMag = corr.transverseMagnetisation( psi, frame )
            magTrans = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( transMag[0] ), 
                vmin=0, vmax=1 )
            fig.colorbar( magTrans )

        case 'MAG_TRANS_ARG':
            transMag = corr.transverseMagnetisation( psi, frame )
            magTrans = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( transMag[1] ), 
                cmap="jet", vmin=-cp.pi, vmax=cp.pi )
            fig.colorbar( magTrans )
        
        case 'GRAD_DENS':
            gradDens = corr.gradient( psi, kwargs['gridSpacing'], frame )
            densGrad = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( abs(gradDens[0])**2 + abs(gradDens[1])**2 ), 
                vmin=0, vmax=1 )
            fig.colorbar( densGrad )
        case 'ABS_MAG':
            transMag = corr.transverseMagnetisation( psi, frame )
            absMag = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( cp.sqrt( abs(transMag[0])**2 +  abs( abs( psi["psi_plus"][ :, :, frame ] ) ** 2 - abs( psi["psi_minus"][ :, :, frame ] ) ** 2 ) )), 
                vmin=0, vmax=2 )
            fig.colorbar( absMag )
            

        case 'ARG_PLUS_DEBUG':
            argPlus = ax.pcolormesh(
                handle_array(psi.grid.x_mesh),
                handle_array(psi.grid.y_mesh),
                handle_array( cp.angle( psi.plus_component[ :, : ] ) ),
                cmap="jet", vmin=-cp.pi, vmax=cp.pi )
            fig.colorbar( argPlus )

        case _:
            raise ValueError(f"{chartType} is not yet implemented.")

    fig.suptitle( prettyString( scalars, titleElements ) )
    plt.savefig(frame_path)

    plt.close()

def takeFrameSpin2( psi, scalars:dict, frames_dir:str, frame:int, chartType:str, titleElements:list[str], groundState:str='FM', **kwargs ) -> None:

    frame_path = f"{frames_dir}/frame_{frame:04d}.png"

    fig, ax = plt.subplots( figsize=(12, 12) )

    try:
        xs = cp.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
        ys = cp.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

        xMesh, yMesh = cp.meshgrid( xs, ys, indexing='ij' )

    except:
        pass
    

    match chartType.upper():
        case 'A20':
            spinSinglet = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( (abs(2*psi['psi_plus2'][:,:,frame]*psi['psi_minus2'][:,:,frame]-2*psi['psi_plus1'][:,:,frame]*psi['psi_minus1'][:,:,frame]+(psi['psi_zero'][:,:,frame])**2)**2)/5 ), 
                vmin=0, vmax=0.2 )
            fig.colorbar( spinSinglet )
        case 'MAGZ':
            magnetisation = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( 2*(abs(psi['psi_plus2'][:,:,frame])**2-abs(psi['psi_minus2'][:,:,frame])**2)+abs(psi['psi_plus1'][:,:,frame])**2-abs(psi['psi_minus1'][:,:,frame])**2), 
                vmin=-2, vmax=2 )
            fig.colorbar( magnetisation )
        case 'MAG':
            mz = 2*(abs(psi['psi_plus2'][:,:,frame])**2-abs(psi['psi_minus2'][:,:,frame])**2)+abs(psi['psi_plus1'][:,:,frame])**2-abs(psi['psi_minus1'][:,:,frame])**2
            mp = 2*( cp.conj(psi['psi_plus1'][:,:,frame])*psi['psi_plus2'][:,:,frame]  + cp.conj(psi['psi_minus2'][:,:,frame])*psi['psi_minus1'][:,:,frame] ) + cp.sqrt(6) * ( cp.conj(psi['psi_zero'][:,:,frame])*psi['psi_plus1'][:,:,frame] + cp.conj(psi['psi_minus1'][:,:,frame])*psi['psi_zero'][:,:,frame] )
            magnetisation = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( cp.sqrt(mz**2 + abs(mp)**2 )), 
                vmin=0, vmax=2 )
            fig.colorbar( magnetisation )
        case 'A30':
            spinTriplet = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array(  abs( 3*cp.sqrt(6) * (psi['psi_minus2'][:,:,frame]*(psi['psi_plus1'][:,:,frame]**2)+ psi['psi_plus2'][:,:,frame]*(psi['psi_minus1'][:,:,frame]**2))/2  + psi['psi_zero'][:,:,frame]*(psi['psi_zero'][:,:,frame]**2-3*psi['psi_plus1'][:,:,frame]*psi['psi_minus1'][:,:,frame]-6*psi['psi_plus2'][:,:,frame]*psi['psi_minus2'][:,:,frame]) )**2 ), 
                vmin=0, vmax=2 )
            fig.colorbar( spinTriplet )
        case 'DENS':
            density = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( abs(psi['psi_plus2'][:,:,frame])**2+ abs(psi['psi_minus2'][:,:,frame])**2 +abs(psi['psi_plus1'][:,:,frame])**2+abs(psi['psi_minus1'][:,:,frame])**2 + abs(psi['psi_zero'][:,:,frame])**2), 
                vmin=0, vmax=1 )
            fig.colorbar( density )
        case 'PHASE2':
            phase = density = ax.pcolormesh(
                handle_array( xMesh ),
                handle_array( yMesh ),
                handle_array( cp.angle(psi['psi_plus2'][:,:,frame])-cp.angle(psi['psi_plus2'][0,0,frame])), 
                vmin=-cp.pi, vmax=cp.pi, cmap='jet' )
            fig.colorbar( phase )
        case _:
            raise ValueError(f"{chartType} is not yet implemented.")

    fig.suptitle( prettyString( scalars, titleElements ) )
    plt.savefig(frame_path)

    plt.close()


def magnetisationQuiverFrame( psi, scalars, frame, frames_dir ):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fx = ( cp.conjugate( psi["psi_plus"][:,:,frame] + psi["psi_minus"][:,:,frame] ) * psi["psi_zero"][:,:,frame] + 
          cp.conjugate( psi["psi_zero"][:,:,frame] ) * ( psi["psi_plus"][:,:,frame] + psi["psi_minus"][:,:,frame] )).real/cp.sqrt(2)
    fy = ( 1j * ( cp.conjugate(psi["psi_minus"][:,:,frame] - psi["psi_plus"][:,:,frame] ) * psi["psi_zero"][:,:,frame] +
                  cp.conjugate( psi["psi_zero"][:,:,frame] ) * ( psi["psi_plus"][:,:,frame] - psi["psi_minus"][:,:,frame] ) )).real/cp.sqrt(2)
    fz = abs(psi["psi_plus"][:,:,frame])**2 - abs(psi["psi_minus"][:,:,frame])**2


    # Create coordinate grids
    xs = cp.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = cp.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    X, Y = cp.meshgrid(xs, ys, indexing='ij')

    fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(18, 12))

    # Subsample to avoid overcrowding (every nth point)
    # 1. Multiple subplots showing different projections

    skip = 4

    # XY projection (colored by fz)
    axs[1][0].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                fx[::skip, ::skip], fy[::skip, ::skip], 
                fz[::skip, ::skip], scale=0.2, scale_units='xy', angles='xy',
                 cmap='RdBu_r')
    axs[1][0].set_title('XY projection (color: $S_z$)')
    axs[1][0].set_aspect('equal')

    # XZ projection (colored by fy)
    axs[1][1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                fx[::skip, ::skip], fz[::skip, ::skip], 
                fy[::skip, ::skip], scale=0.2, scale_units='xy', angles='xy',
                 cmap='RdBu_r')
    axs[1][1].set_title('XZ projection (color: $S_y$)')
    axs[1][1].set_aspect('equal')

    # YZ projection (colored by fx)
    axs[1][2].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                fy[::skip, ::skip], fz[::skip, ::skip], 
                fx[::skip, ::skip], scale=0.2, scale_units='xy', angles='xy',
                 cmap='RdBu_r')
    axs[1][2].set_title('YZ projection (color: $S_x$)')
    axs[1][2].set_aspect('equal')


    # 2. Color by spin direction (spherical angles)
    # This is great for seeing domain structure
    theta = cp.arccos(fz / (cp.sqrt(fx**2 + fy**2 + fz**2) + 1e-10))  # polar angle
    phi = cp.arctan2(fy, fx)  # azimuthal angle

    # Color by azimuthal angle (direction in xy-plane)
    Q1 = axs[0][1].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                        fx[::skip, ::skip], fy[::skip, ::skip],
                        phi[::skip, ::skip], scale=0.2, scale_units='xy', angles='xy',
                         cmap='hsv', 
                        clim=(-cp.pi, cp.pi))
    axs[0][1].set_title('Spin field (color: azimuthal angle)')
    axs[0][1].set_aspect('equal')
    plt.colorbar(Q1, ax=axs[0][1], label='$\\phi$')

    # Color by polar angle (tilt from z-axis)
    Q2 = axs[0][2].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                        fx[::skip, ::skip], fy[::skip, ::skip],
                        theta[::skip, ::skip], scale=0.2, scale_units='xy', angles='xy',
                         cmap='viridis',clim=(0,cp.pi))
    axs[0][2].set_title('Spin field (color: polar angle)')
    axs[0][2].set_aspect('equal')
    plt.colorbar(Q2, ax=axs[0][2], label='$\\theta$')

    # 3. Overlay on spin magnitude or density

    # Background: spin magnitude
    S_mag = cp.sqrt(fx**2 + fy**2 + fz**2)
    im = axs[0][0].imshow(S_mag.T, extent=[xs[0], xs[-1], ys[0], ys[-1]], 
                origin='lower', cmap='gray', alpha=0.6,clim=(0,2))

    # Overlay: spin direction
    Q = axs[0][0].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                fx[::skip, ::skip], fy[::skip, ::skip],
                scale=0.2, scale_units='xy', angles='xy',
                color='red', alpha=0.8, width=0.003)

    axs[0][0].set_xlabel('x')
    axs[0][0].set_ylabel('y')
    axs[0][0].set_title('Spin texture on |S| background')
    axs[0][0].set_aspect('equal')
    plt.colorbar(im, ax=axs[0][0], label='|S|')

    plt.tight_layout()
     
    plt.savefig(frame_path)

    plt.close()



def superfluidVelocitiesFrame(  psi, scalars, frame, frames_dir ):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    skip = 4
    psiPlusX, psiPlusY = cp.gradient( psi['psi_plus'][:,:,frame], scalars['dx'], scalars['dy'] )
    psiZeroX, psiZeroY = cp.gradient( psi['psi_zero'][:,:,frame], scalars['dx'], scalars['dy'] )
    psiMinusX, psiMinusY = cp.gradient( psi['psi_minus'][:,:,frame], scalars['dx'], scalars['dy'] )
    vxprecursor = cp.conj(psi['psi_plus'][:,:,frame]) * psiPlusX + cp.conj(psi['psi_zero'][:,:,frame]) * psiZeroX + cp.conj(psi['psi_minus'][:,:,frame]) * psiMinusX 
    vyprecursor = cp.conj(psi['psi_plus'][:,:,frame]) * psiPlusY + cp.conj(psi['psi_zero'][:,:,frame]) * psiZeroY + cp.conj(psi['psi_minus'][:,:,frame]) * psiMinusY 
    vx = (vxprecursor - cp.conj(vxprecursor)).imag
    vy = (vyprecursor - cp.conj(vyprecursor)).imag

    vfzxprecursor = cp.conj(psi['psi_plus'][:,:,frame]) * psiPlusX - cp.conj(psi['psi_minus'][:,:,frame]) * psiMinusX 
    vfzyprecursor = cp.conj(psi['psi_plus'][:,:,frame]) * psiPlusY - cp.conj(psi['psi_minus'][:,:,frame]) * psiMinusY 
    vfzx = (vfzxprecursor - cp.conj(vfzxprecursor)).imag
    vfzy = (vfzyprecursor - cp.conj(vfzyprecursor)).imag
    
    xs = cp.arange( -scalars['nx']//2, scalars['nx']//2     ) * scalars['dx']   
    ys = cp.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    X, Y = cp.meshgrid(xs, ys, indexing='ij')
    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
    axs[0].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                vx[::skip, ::skip], vy[::skip, ::skip] )
    axs[0].set_title('Mass Flow')
    axs[1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                vfzx[::skip, ::skip], vfzy[::skip, ::skip] )
    axs[1].set_title('Mag Z Flow')
    plt.savefig(frame_path)
    plt.close()



def allComponentFrame(psi, scalars, frame, frames_dir):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(18,6))
    xs = cp.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = cp.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    xMesh, yMesh = cp.meshgrid( xs, ys, indexing='ij' )

    densityPlus = axs[0].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_plus"][ :, :, frame ] ) ) ** 2,
        vmin=0, vmax=1.2 )
    axs[0].set_aspect('equal')
    fig.colorbar( densityPlus )

    densityZero = axs[1].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_zero"][ :, :, frame ] ) ) ** 2,
        vmin=0, vmax=1.2 )
    axs[1].set_aspect('equal')
    fig.colorbar( densityZero )

    densityMinus = axs[2].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_minus"][ :, :, frame ] ) ) ** 2,
        vmin=0, vmax=1.2 )
    axs[2].set_aspect('equal')
    fig.colorbar( densityMinus )

    plt.savefig(frame_path)

    plt.close()


def allArgsFrame(psi, scalars, frame, frames_dir):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(18,6))
    xs = cp.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = cp.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    xMesh, yMesh = cp.meshgrid( xs, ys, indexing='ij' )

    argPlus = axs[0].pcolormesh(
        (xMesh),
        (yMesh),
        ( cp.angle( psi["psi_plus"][ :, :, frame ] ) - cp.angle( psi["psi_plus"][0,0,frame])) % (2*cp.pi),
        vmin=0, vmax=2*cp.pi, cmap='jet' )
    axs[0].set_aspect('equal')
    fig.colorbar( argPlus )

    argZero = axs[1].pcolormesh(
        (xMesh),
        (yMesh),
        (cp.angle( psi["psi_zero"][ :, :, frame ] )- cp.angle( psi["psi_zero"][0,0,frame]))% (2*cp.pi),
        vmin=0, vmax=2*cp.pi, cmap='jet' )
    axs[1].set_aspect('equal')
    fig.colorbar( argZero )

    argMinus = axs[2].pcolormesh(
        (xMesh),
        (yMesh),
        (cp.angle( psi["psi_minus"][ :, :, frame ] )- cp.angle( psi["psi_minus"][0,0,frame]))% (2*cp.pi),
        vmin=0, vmax=2*cp.pi , cmap='jet')
    axs[2].set_aspect('equal')
    fig.colorbar( argMinus )

    plt.savefig(frame_path)

    plt.close()

def allComponentSpin2Frame(psi, scalars, frame, frames_dir):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(18,12))
    xs = cp.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = cp.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    xMesh, yMesh = cp.meshgrid( xs, ys, indexing='ij' )

    densityPlus2 = axs[0][0].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_plus2"][ :, :, frame ] ) ) ** 2,
        vmin=0, vmax=1.2 )
    axs[0][0].set_aspect('equal')
    fig.colorbar( densityPlus2 )

    densityZero = axs[0][1].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_zero"][ :, :, frame ] ) ) ** 2,
        vmin=0, vmax=1.2 )
    axs[0][1].set_aspect('equal')
    fig.colorbar( densityZero )

    densityMinus2 = axs[0][2].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_minus2"][ :, :, frame ] ) ) ** 2,
        vmin=0, vmax=1.2 )
    axs[0][2].set_aspect('equal')
    fig.colorbar( densityMinus2 )

    densityPlus1 = axs[1][0].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_plus1"][ :, :, frame ] ) ) ** 2,
        vmin=0, vmax=1.2 )
    axs[1][0].set_aspect('equal')
    fig.colorbar( densityPlus1 )

    densityMinus1 = axs[1][2].pcolormesh(
        (xMesh),
        (yMesh),
        abs(( psi["psi_minus1"][ :, :, frame ] ) ) ** 2,
        vmin=0, vmax=1.2 )
    axs[1][2].set_aspect('equal')
    fig.colorbar( densityMinus1 )

    plt.savefig(frame_path)

    plt.close()

def allArgsChemPotFrame(psi, scalars, frame, frames_dir):
    frame_path = f"{frames_dir}/frame_{frame:04d}.png"
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(18,6))
    xs = cp.arange( -scalars['nx']//2, scalars['nx']//2 ) * scalars['dx']   
    ys = cp.arange( -scalars['ny']//2, scalars['ny']//2 ) * scalars['dy']   

    xMesh, yMesh = cp.meshgrid( xs, ys, indexing='ij' )

    argPlus = axs[0].pcolormesh(
        (xMesh),
        (yMesh),
        cp.angle( psi["psi_plus"][ :, :, frame ] * cp.exp(1j*scalars['dt']*scalars['frameRate']*frame*(scalars['c0']+scalars['c2'] ) ) ) ,
        vmin=-cp.pi, vmax=cp.pi, cmap='jet' )
    axs[0].set_aspect('equal')
    fig.colorbar( argPlus )

    argZero = axs[1].pcolormesh(
        (xMesh),
        (yMesh),
        cp.angle( psi["psi_zero"][ :, :, frame ]*cp.exp(1j*scalars['dt']*scalars['frameRate']*frame*(scalars['c0']+scalars['c2'] ) ) ),
        vmin=-cp.pi, vmax=cp.pi, cmap='jet' )
    axs[1].set_aspect('equal')
    fig.colorbar( argZero )

    argMinus = axs[2].pcolormesh(
        (xMesh),
        (yMesh),
        cp.angle( psi["psi_minus"][ :, :, frame ]*cp.exp(1j*scalars['dt']*scalars['frameRate']*frame*(scalars['c0']+scalars['c2'] ) ) ),
        vmin=-cp.pi, vmax=cp.pi , cmap='jet')
    axs[2].set_aspect('equal')
    fig.colorbar( argMinus )

    plt.savefig(frame_path)

    plt.close()

def movieFromFrames( movieName:str, frames_dir:str ) -> None:
    
    def strFunc( frame:str ) -> int:
        values = str( frame ).split('_')
        return int( values[ -1 ][ :-4 ]) # the :-4 is removing .png from the end of the string

    frames = []
    for filename in Path(frames_dir).iterdir():
        frames.append( filename )
    frames.sort( key=strFunc )
    with imageio.get_writer( movieName, fps=10 ) as writer:
        for filename in frames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up frames
    import shutil

    shutil.rmtree(frames_dir)


def createFrames( waveFunc, scalars:dict, framesDir:str, chartType:str, titleElements:list[str], groundState:str='FM', spin=1, **kwargs ):
    os.makedirs(framesDir, exist_ok=True)
    if spin == 1:
        for i in range( scalars["nt"]//scalars["frameRate"] ):
            takeFrame( waveFunc, scalars, framesDir, i, chartType, titleElements, groundState, gridSpacing=kwargs['gridSpacing'] )
    elif spin == 2:
        for i in range( scalars["nt"]//scalars["frameRate"] ):
            takeFrameSpin2( waveFunc, scalars, framesDir, i, chartType, titleElements, groundState, gridSpacing=kwargs['gridSpacing'] )

def createMovie( waveFunc, scalars:dict, movieName:str, chartType:str, titleElements:list[str], groundState:str, **kwargs ) -> None:
    framesDir = 'frames'
    createFrames(waveFunc, scalars, framesDir, chartType, titleElements, groundState, gridSpacing=kwargs['gridSpacing'] )
    movieFromFrames( movieName, framesDir )



if __name__ == '__main__':
    movieFromFrames( 'tau_q=400TestGRADDENS.mp4', 'frames' )
    