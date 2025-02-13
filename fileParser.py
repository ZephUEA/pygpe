import h5py
import numpy as np
import matplotlib.pyplot as plt
import correlation as corr

if __name__ == '__main__':
    domainMaxes = []
    transitionTimes = []
    transitionCorrelations = []
    choices = range( 0, 55, 5 )
    quenchTimes = []
    for integer in choices:
        allLengths = []
        allTimes = []
        with h5py.File( f'./data/scratch/domains{integer**2}.hdf5', 'r' ) as file:
            quenchEndtime = integer**2
            for run in file.keys():
                domains = np.array( file[run]['Domains'][()])
                correlator = np.array( file[run]['Correlations'][()] )
                radii = np.array( file[run]['Radii'] )
                times = np.array( file[run]['Times'][()] )
                firstZeros = np.zeros( times.shape )
                for index in range( times.shape[0] ):
                    firstZeros[index] = corr.firstZero( correlator[index,:] )
                transitionIndex = max( enumerate( domains ), key=lambda x: x[1])[0]
                domainMaxes.append( max( domains ) )
                transitionTimes.append( times[transitionIndex] )
                transitionCorrelations.append( radii[ corr.firstZero( correlator[transitionIndex,:] ) ] )
                quenchTimes.append( integer ** 2 )
                lengths = []
                for time in range(len(times)):
                    lengths.append( radii[ corr.firstZero( correlator[time,:] ) ] )
                plt.loglog( times[transitionIndex:], lengths[transitionIndex:], label='_nolegend_' )
                allTimes += list( times[transitionIndex+30:] )
                allLengths += list( lengths[transitionIndex+30:] )
            plt.axvline( 0, color='blue' )
            plt.axvline( times[transitionIndex], color='red' )
            plt.axvline( quenchEndtime, color='green' )
            ts = np.array( allTimes )
            b,m = corr.bestFitLine( np.log10( ts ), np.log10( np.array( allLengths ) ) )
            r2 = corr.r2( np.log10( ts ), np.log10( np.array( allLengths ) ) )
            tprime = np.linspace( min( ts ), max( ts ), 1000 )
            # plt.loglog( tprime, 10**b * tprime**m, color='black' )
            plt.xlabel( 'T - tau' )
            plt.ylabel( 'L(t)' )
            plt.title( f'quench time = {integer**2}' )
            plt.legend(['Transition','Domain Formation', 'End of Quench',f'{10**b:.2f} * (t-tau)^{m:.4f}, R^2={r2:.4f}'])
            plt.show()
            plt.cla()

    # The first 3 data points do not follow the KZM scaling laws due to the quench being too fast
    plt.scatter( quenchTimes[151:],  transitionTimes[151:]  )
    b,m = corr.bestFitLine( np.log10(quenchTimes[151:]), np.log10(transitionTimes[151:]) )
    r2 = corr.r2( np.log10(quenchTimes[151:]), np.log10(transitionTimes[151:]) )
    plt.plot( quenchTimes[151:], 10**b * quenchTimes[151:]**m )
    plt.title('Freezing time vs tau_Q')
    plt.xlabel('tau_Q') 
    plt.ylabel('Freezing time')
    plt.legend(['data',f'{10**b:.2f} * tau^{m:.4f}, R^2={r2:.4f}'])
    plt.show()
            
    # with h5py.File( './data/kzm_2d_t_instant.hdf5' ) as file:
    #     psi = file['wavefunction']
    #     scalars = {
    #         "c0": 10,
    #         "c2": -0.5,
    #         "p": 0,
    #         "q": 0.5,
    #         "n0": 1,
    #         "trap":0.0,
    #         # Time params
    #         "dt": (1) * 1e-2,
    #         "dq": -1,
    #         "nt": 600_000,
    #         "t": 0,
    #         "tau_q":0,
    #         "frameRate":1000,
    #         "Q_0" : 1.0,
    #         "nx" : 512,
    #         "ny" : 512,
    #         "dx" : 0.5,
    #         "dy" : 0.5
    #         }
    #     correlation = corr.correlatorFM( psi, scalars )
    #     ( xs, averageCorrelation ) = corr.radialAverage( correlation, scalars )
    #     firstZeros = np.zeros( (600,) )
    #     for index in range( len( firstZeros ) ):
    #         firstZeros[index] = corr.firstZero( averageCorrelation[:,index] )
    #     ts = [i for i in range(600)][8:]
    #     b,m = corr.bestFitLine( np.log10(ts), np.log10(firstZeros[8:]) )
    #     plt.loglog( ts, firstZeros[8:] )
    #     plt.loglog( ts, 10**b * ts**m )
    #     plt.legend(['data',f'b={b:.2f}, m={m:.2f}'])
    #     plt.show()
    #     print( f'b={b}, m={m}' )