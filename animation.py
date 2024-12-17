import imageio.v2 as imageio
from pathlib import Path
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Applications/ffmpeg"
import matplotlib.pyplot as plt
from pygpe.shared.utils import handle_array
try:
    import cupy as cp  # type: ignore
except ImportError:
    import numpy as cp

def prettyString( params:dict, show:list[str] ) -> str:
    string = f''
    for key,value in params.items():
        if key not in show:
            continue
        string += f'{key} = {value:.3f}, '
    return string[:-2]

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
        


def takeFrame( psi, scalars:dict, frames_dir:str, frame:int, chartType:str, titleElements:list[str], groundState:str='FM' ) -> None:

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


def createFrames( waveFunc, scalars:dict, framesDir:str, chartType:str, titleElements:list[str], groundState:str='FM' ):
    os.makedirs(framesDir, exist_ok=True)
    for i in range( scalars["nt"]//scalars["frameRate"] ):
        takeFrame( waveFunc, scalars, framesDir, i, chartType, titleElements, groundState )

def createMovie( waveFunc, scalars:dict, movieName:str, chartType:str, titleElements:list[str], groundState:str ) -> None:
    framesDir = 'frames'
    createFrames(waveFunc, scalars, framesDir, chartType, titleElements, groundState )
    movieFromFrames( movieName, framesDir )



if __name__ == '__main__':
    movieFromFrames( 'kzm2dt1000.mp4', 'frames' )
    