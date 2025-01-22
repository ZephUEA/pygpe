import numpy as np

def parseDomainsFile( filename ):
    domainsList = []
    timesList = []
    with open( filename, 'r' ) as file:
        times = []
        for line in file.readlines():
            values = line.split( ',' )
            parsedValues = list( map( lambda x: float( x.strip() ), filter( lambda y: y != '\n', values ) ) )
            if len( parsedValues ) <= 20:
                times += parsedValues
            else:
                timesList.append( times )
                times = []
                domainsList.append( parsedValues )
    return ( timesList, domainsList )

if __name__ == '__main__':
    times, domains = parseDomainsFile( './data/domains225.txt' )
    for index in range( len( times ) ):
        # print( len( times[ index ]), len(domains[ index ]))
        indexMax, domainMax = max( enumerate( domains[index] ), key=lambda x: x[1] ) 
        time = times[index][indexMax]
        print( time, domainMax )
    maxDomains = list( map( lambda x: max(x), domains ) )
    print( sum(maxDomains) / len( maxDomains ) ) 