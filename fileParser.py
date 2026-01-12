import h5py
import numpy as np
import matplotlib.pyplot as plt
import correlation as corr
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def domainSize( radii, correlator ):
    domainsize = []
    for i in range( correlator.shape[0] ):
        try:
            domainsize.append( radii[ corr.firstZero( correlator[i,:] ) ] )
        except ValueError:
            return np.array(domainsize)
    return np.array(domainsize)


if __name__ == '__main__':
    isPolar = True
    coarsening = True
    if isPolar:
        fileString = './data/results2/baLiPolarDomains'
        maxRange = 60
        chartString = 'Polar'
    else: 
        fileString = './data/results3/baLiDomains'
        maxRange = 65
        chartString = ''
    
    if coarsening:
        lowerbound = 5
    else:
        lowerbound = 15

    domainMaxes = []
    transitionTimes = []
    transitionCorrelations = []
    correlationFunctions = []
    lateCorrelations = []
    choices = range( lowerbound, maxRange, 5 )
    quenchTimes = []
    systematicCorrelations = None
    timeFlag = True
    shortQuenchLengths = []
    longQuenchLengths = []
    legends = []
    fig, axs = plt.subplots(nrows=2,sharex=True, figsize=(6,6))
    for integer in choices:
        allLengths = []
        allTimes = []
        with h5py.File( fileString + f'{integer**2}.hdf5', 'r' ) as file:
            quenchTime = integer**2
            domainSum = None
            dom25 = None
            dom400 = None
            dom1600 = None
            dom3600 = None
            for run in file.keys():
                domains = np.array( file[run]['Domains'][()])
                if domainSum is None:
                    domainSum = domains
                else:
                    minlength = min( domainSum.shape[0], domains.shape[0] )
                    domainSum = domainSum[:minlength] + domains[:minlength]
                correlator = np.array( file[run]['Correlations'][()] )
                radii = np.array( file[run]['Radii'] )
                if isPolar:
                    times = np.array( file[run]['Times'][()] ) - ( 2 * integer ** 2 )
                else:
                    times = np.array( file[run]['Times'][()] )
                # This is for the coarsening information
                transitionIndex = max( enumerate( domains ), key=lambda x: x[1])[0]
                
                domainMaxes.append( max( domains ) )
                transitionTimes.append( times[transitionIndex] )
                zeroIndex = corr.firstZero( correlator[transitionIndex,:] )
                zeroPosition = ( correlator[transitionIndex,zeroIndex-1] * (radii[zeroIndex]-radii[zeroIndex-1]) / (correlator[transitionIndex,zeroIndex-1]-correlator[transitionIndex,zeroIndex]) ) + radii[zeroIndex-1]

                transitionCorrelations.append( zeroPosition ) # zeroPosition is a linear interpolation



                correlationFunctions.append( correlator[transitionIndex,:] )
                lateCorrelations.append( correlator[min( 2*transitionIndex,correlator.shape[0]-1), :])
                if quenchTime == 400 and systematicCorrelations is None:
                    systematicCorrelations = [ correlator[index,:] for index in range( transitionIndex+10, transitionIndex + 110, 10 )]
                    testCorrelations = [ correlator[index,:] for index in range( transitionIndex-30, transitionIndex, 5 )]
                quenchTimes.append( integer ** 2 )

                if quenchTime == 25:
                    domainsizes = domainSize( radii, correlator )
                    if dom25 is None:
                        dom25 = domainsizes
                        dom25Len = 1
                    else:
                        minlength = min( dom25.shape[0], domainsizes.shape[0] )
                        dom25 = dom25[:minlength] + domainsizes[:minlength]
                        dom25Len += 1
                
                if quenchTime == 400:
                    domainsizes = domainSize( radii, correlator )
                    if dom400 is None:
                        dom400 = domainsizes
                        dom400Len = 1
                    else:
                        minlength = min( dom400.shape[0], domainsizes.shape[0] )
                        dom400 = dom400[:minlength] + domainsizes[:minlength]
                        dom400Len += 1
                
                # if quenchTime == 1600:
                #     domainsizes = domainSize( radii, correlator )
                #     if dom1600 is None:
                #         dom1600 = domainsizes
                #         dom1600Len = 1
                #     else:
                #         minlength = min( dom1600.shape[0], domainsizes.shape[0] )
                #         dom1600 = dom1600[:minlength] + domainsizes[:minlength]
                #         dom1600Len += 1

                # if quenchTime == 3600:
                #     domainsizes = domainSize( radii, correlator )
                #     if dom3600 is None:
                #         dom3600 = domainsizes
                #         dom3600Len = 1
                #     else:
                #         minlength = min( dom3600.shape[0], domainsizes.shape[0] )
                #         dom3600 = dom3600[:minlength] + domainsizes[:minlength]
                #         dom3600Len += 1
            offset=20
            if timeFlag:
                axs[1].loglog( times[transitionIndex+offset:], 7*10**5 * times[transitionIndex+offset:]**(-4/3), color='black' )
                timeFlag = False
            if integer in [5,20,40,60]:
                axs[1].loglog( times[transitionIndex+offset:len(domainSum)],  domainSum[transitionIndex+offset:] / len( file.keys() ) )
                legends.append( rf'$\tau_Q={quenchTime}$')

        if dom25 is not None:
            axs[0].loglog( times[transitionIndex+offset:], 0.32*times[transitionIndex+offset:]**(2/3), color='black' )
            axs[0].loglog( times[transitionIndex+offset:len(dom25)], dom25[transitionIndex+offset:] / dom25Len ) # Need to divide by the number of runs
        if dom400 is not None:
            axs[0].loglog( times[transitionIndex+offset:len(dom400)], dom400[transitionIndex+offset:] / dom400Len)
        # if dom1600 is not None:
        #     axs[0].loglog( times[transitionIndex+offset:len(dom1600)], dom1600[transitionIndex+offset:] / dom1600Len )
        # if dom3600 is not None:
        #     axs[0].loglog( times[transitionIndex+offset:len(dom3600)], dom3600[transitionIndex+offset:] / dom3600Len )

            


    axs[0].legend([r'$(t-\tau_Q)^{2/3}$', r'$\tau_Q=25$', r'$\tau_Q=400$',r'$\tau_Q=1600$',r'$\tau_Q=3600$'])   
    axs[1].legend( [r'$(t-\tau_Q)^{-4/3}$'] + legends )
    axs[0].yaxis.set_minor_formatter(ScalarFormatter())

    labels = ['a)','b)']
    for index, ax in enumerate(axs.flat):
        ax.annotate(labels[index], xy=(0, 1), xytext=(-20, 10), 
                xycoords='axes fraction', textcoords='offset points',
                fontsize=12, fontweight='bold', ha='left', va='bottom')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1, left=0.15)
    fig.text(0.5, 0.04, r'$t-\tau_Q$', ha='center')
    fig.text(0.04, 0.75, r'$L(t)$', va='center', rotation='vertical')
    fig.text(0.04, 0.25, r'$\rho$', va='center', rotation='vertical')
    # plt.show()
    plt.savefig( f'./paperCharts/coarseningPolar.png' )
    plt.cla()


            #     lengths = []
            #     exitIndex = None
            #     for time in range(len(times)):
            #         try:
            #             lengths.append( radii[ corr.firstZero( correlator[time,:] ) ] )
            #         except ValueError:
            #             exitIndex = time
            #             break
            #     if exitIndex:
            #         times = times[:exitIndex]
            #     offset = 70
            #     plt.loglog( times[transitionIndex+offset:], lengths[transitionIndex+offset:], label='_nolegend_' )
            #     allTimes += list( times[transitionIndex+offset:] )
            #     allLengths += list( lengths[transitionIndex+offset:] )
            
            # # plt.axvline( 0, color='blue' )
            # # plt.axvline( times[transitionIndex], color='red' )
            # # plt.axvline( 2 * quenchTime, color='green' )
            # ts = np.array( allTimes )
            # xs = np.array( allLengths )
            # b,m = corr.bestFitLine( np.log10( ts ), np.log10( xs ) )
            # r2 = corr.r2( np.log10( ts ), np.log10( xs ) )
            # tprime = np.linspace( min( ts ), max( ts ), 1000 )
            # plt.loglog( tprime, 10**b * tprime**m, color='black' )
            # plt.xlabel( r'$t - \tau_Q$' )
            # plt.ylabel( r'$L(t)$' )
            # plt.title( fr'First Zero of the correlation function versus time for $\tau_Q = {integer**2}$' )
            # # plt.legend(['Transition','Domain Formation', 'End of Quench',fr'${10**b:.2f} * (t-\tau_Q)^{{{m:.4f}}}, R^2={r2:.4f}$'])
            # plt.legend([fr'${10**b:.2f} * (t-\tau_Q)^{{{m:.4f}}}, R^2={r2:.4f}$'])

            # # sub_ax = inset_axes(
            # #     parent_axes=ax,
            # #     width="40%",
            # #     height="30%",
            # #     borderpad=1  # padding between parent and inset axes
            # #     )
            
            # # for run in file.keys():
            # #     if isPolar:
            # #         times = np.array( file[run]['Times'][()] ) - ( 2 * integer ** 2 )
            # #     else:
            # #         times = np.array( file[run]['Times'][()] )
            # #     lengths = []
            # #     exitIndex = None
            # #     for time in range(len(times)):
            # #         try:
            # #             lengths.append( radii[ corr.firstZero( correlator[time,:] ) ] )
            # #         except ValueError:
            # #             exitIndex = time
            # #             break
            # #     if exitIndex:
            # #         times = times[:exitIndex]
            # #     plt.plot( times, lengths, marker='.', markersize='1', label='_nolegend_' )
            
            # plt.show()
            # # # plt.savefig(f'./paperCharts/linearCoarsening{integer**2}.png')
            # plt.cla()

   
   
   
   
   
   

    transitionTimes = np.array(transitionTimes)
    changeIndices = list( { quenchTimes.index(i) for i in quenchTimes } )
    changeIndices.sort()
    quenchTimeUnique = list( set(quenchTimes) )
    quenchTimeUnique.sort()
    meanTransitionTimes = [ sum( transitionTimes[changeIndices[i]:changeIndices[i+1]] ) / (changeIndices[i+1]-changeIndices[i])  for i in range(len(changeIndices)-1)] + [sum( transitionTimes[changeIndices[-1]:] ) / (changeIndices[-1]-changeIndices[-2])]
    
    
    # #KZM Plots
    # fig, axs = plt.subplots(nrows=2,figsize=(6,6))
    # axs[0].errorbar( quenchTimeUnique,  meanTransitionTimes, yerr=5, fmt='o', mfc='none' )
    # b,m = corr.bestFitCurveError( lambda x, b, m :b * x ** m, quenchTimes, transitionTimes, 5 ) 

    # # results = corr.power_law_regression_logspace( quenchTimes, transitionTimes / 1, 5 ) 
    # # print( m )
    # # print( f'{results['m']} +- {results['m_err']}' )
    # # print( f'{results['b']} +- {results['b_err']}' )
    # axs[0].loglog( quenchTimes, b[0] * np.array(quenchTimes)**m[0] )
    # axs[0].set_ylabel('Freezing time')
    # # plt.loglog( quenchTimes, 10.6 * np.array(quenchTimes)**0.5 )
    # axs[0].legend([fr'$T_0\tau_Q^{{{m[0]:.4f}}}$','data'])
    # axs[0].yaxis.set_minor_formatter(ScalarFormatter())

    # domainMaxes = np.array( domainMaxes )
    # meanDomainMaxes = [ sum( domainMaxes[changeIndices[i]:changeIndices[i+1]] ) / (changeIndices[i+1]-changeIndices[i])  for i in range(len(changeIndices)-1)] + [sum( domainMaxes[changeIndices[-1]:] ) / (changeIndices[-1]-changeIndices[-2])]
    # maxDomainMaxes = [ max( domainMaxes[changeIndices[i]:changeIndices[i+1]] ) - meanDomainMaxes[i]  for i in range(len(changeIndices)-1)] + [ max( domainMaxes[changeIndices[-1]:] ) - meanDomainMaxes[-1] ]
    # minDomainMaxes = [ meanDomainMaxes[i] - min( domainMaxes[changeIndices[i]:changeIndices[i+1]] ) for i in range(len(changeIndices)-1)] + [ meanDomainMaxes[-1] - min( domainMaxes[changeIndices[-1]:] ) ]
    # errorDomainMaxes = np.array([minDomainMaxes,maxDomainMaxes])
    # stdDevDomainMaxes = [ np.std( domainMaxes[changeIndices[i]:changeIndices[i+1]] )  for i in range(len(changeIndices)-1)] + [np.std( domainMaxes[changeIndices[-1]:] ) ]
    # axs[1].errorbar( quenchTimeUnique, meanDomainMaxes, yerr=errorDomainMaxes, fmt='o', mfc='none' )
    # bDom, mDom = corr.bestFitCurveError( lambda x, b, m : b * x ** m, quenchTimes, domainMaxes, None )
    # axs[1].loglog( quenchTimes, bDom[0] * quenchTimes**mDom[0] )
    # axs[1].set_xlabel(r'$\tau_Q$')
    # axs[1].set_ylabel('Domains at transition')
    # axs[1].legend([fr'$\rho_0\tau_Q^{{{mDom[0]:.4f}}}$','data'])

    
    # axs[1].yaxis.set_minor_formatter(ScalarFormatter())
    # labels = ['a)','b)']
    # for index, ax in enumerate(axs.flat):
    #     ax.annotate(labels[index], xy=(0, 1), xytext=(-20, 10), 
    #             xycoords='axes fraction', textcoords='offset points',
    #             fontsize=12, fontweight='bold', ha='left', va='bottom')
    # plt.tight_layout()
    
    # plt.savefig('./paperCharts/kzm' + chartString + 'MeasuresLogLog.png')
    # # plt.show()
    # plt.cla()


    # fig, axs = plt.subplots(nrows=2,figsize=(6,6))
    # meanTransitionCorrelations = [ sum( transitionCorrelations[changeIndices[i]:changeIndices[i+1]] ) / (changeIndices[i+1]-changeIndices[i])  for i in range(len(changeIndices)-1)] + [sum( transitionCorrelations[changeIndices[-1]:] ) / (changeIndices[-1]-changeIndices[-2])]
    # maxTransitionCorrelations = [ max( transitionCorrelations[changeIndices[i]:changeIndices[i+1]] ) - meanTransitionCorrelations[i]  for i in range(len(changeIndices)-1)] + [ max( transitionCorrelations[changeIndices[-1]:] ) - meanTransitionCorrelations[-1] ]
    # minTransitionCorrelations = [ meanTransitionCorrelations[i] - min( transitionCorrelations[changeIndices[i]:changeIndices[i+1]] ) for i in range(len(changeIndices)-1)] + [ meanTransitionCorrelations[-1] - min( transitionCorrelations[changeIndices[-1]:] ) ]
    # errorTransitionCorrelations = np.array([minTransitionCorrelations,maxTransitionCorrelations])
    # stdDevTransitionCorrelations = [ np.std( transitionCorrelations[changeIndices[i]:changeIndices[i+1]] )  for i in range(len(changeIndices)-1)] + [np.std( transitionCorrelations[changeIndices[-1]:] ) ]
    
    
    # axs[1].errorbar( quenchTimeUnique, meanTransitionCorrelations, yerr=errorTransitionCorrelations, fmt='o', mfc='none' )
    # bTrans, mTrans = corr.bestFitCurveError( lambda x, b, m : b * ( x ** m ), quenchTimes, transitionCorrelations, None ) 
    # axs[1].loglog( quenchTimes, bTrans[0] * quenchTimes**mTrans[0] )
    # axs[1].set_xlabel(r'$\tau_Q$')
    # axs[1].set_ylabel('First zero radius')
    # axs[1].legend([fr'$L_0\tau_Q^{{{mTrans[0]:.4f}}}$','data'])
    # axs[1].yaxis.set_minor_formatter(ScalarFormatter())
    # axs[1].yaxis.set_major_formatter(ScalarFormatter())

    # size = min( map( lambda x: x.shape[0], correlationFunctions ) )
    # newCorrelator = np.zeros( size )
    # for index, correlator in enumerate( correlationFunctions ):
    #     newCorrelator += correlator
    #     # if index % 10 != 0:
    #     #     continue
    #     # axs[0].plot( radii, correlator/correlator[0] )
    #     if index % 10 == 9:
    #         axs[0].plot( radii, newCorrelator/newCorrelator[0] )
    #         newCorrelator = np.zeros( size )

    # axs[0].set_xlabel(r'$Radius$')
    # axs[0].set_ylabel('Correlation')
    # axs[0].axhline(xmax=max(radii), color='k')


    # labels = ['a)','b)']
    # for index, ax in enumerate(axs.flat):
    #     ax.annotate(labels[index], xy=(0, 1), xytext=(-20, 10), 
    #             xycoords='axes fraction', textcoords='offset points',
    #             fontsize=12, fontweight='bold', ha='left', va='bottom')
    # plt.tight_layout()

    # plt.savefig('./paperCharts/kzm' + chartString + 'CorrelationFunctions.png')
    # # plt.show()
    # plt.cla()


    # fig, ax = plt.subplots()
    
    # legend = []
    # for correlator in systematicCorrelations :
    #     ax.plot( radii, correlator/correlator[0] )
    # ax.axhline(xmax=max(radii), color='k')
    # plt.xlabel('Radius')
    # plt.ylabel('Correlation')
    # sub_ax = inset_axes(
    # parent_axes=ax,
    # width="40%",
    # height="30%",
    # borderpad=1  # padding between parent and inset axes
    # )
    # for correlator in systematicCorrelations:
    #     sub_ax.plot( radii / radii[corr.firstZero(correlator)], correlator/(correlator[0]) )
    # sub_ax.axhline(xmax=max(radii), color='k')
    # plt.xlabel(r'$R/L(t)$')
    # plt.ylabel('Correlation')
    # plt.savefig('./paperCharts/correlatorAfterTransition.png')
    # # plt.show()
    # plt.cla()
    

    # for correlator in testCorrelations:
    #     plt.plot( radii , correlator/(correlator[0]) )
    # plt.axhline(xmax=max(radii), color='k')
    # plt.xlabel(r'$R$')
    # plt.ylabel('Correlation')
    # # plt.savefig('./paperCharts/correlatorAtTransition.png')
    # plt.show()
    # plt.cla()

    # legend = []
    # for index,function in enumerate( lateCorrelations ):
    #     if index % 10 != 0:
    #         continue
    #     plt.plot( radii, function/ function[0] )
    #     legend.append(fr'$\tau_q={quenchTimes[index]}$')
    # plt.axhline(xmax=max(radii))
    # plt.title(fr'Correlation function long after transition for $\tau_Q = $ {quenchTimes[index]}')
    # plt.xlabel('Radius')
    # plt.ylabel('Correlation')
    # plt.legend(legend)
    # # plt.show()

            