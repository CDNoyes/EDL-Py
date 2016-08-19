
def writeSummary(summaryData):
    
    headLine = '#Summary File'
    stateLine = ''
    valLine = ''
    unitLine = ''
    length = 20
    for state,unit,val in summaryData:
        stateLine += (state + ' '*(length-len(state)))
        valLine += (str(val)+' '*(length-len(str(val))))
        unitLine += (unit + ' '*(length-len(unit)))
    
    
    with open('./results/summary.dat','w') as summary:
        for line in [headLine,stateLine,unitLine,valLine]:
            summary.write(line+'\n')



def writeEvents():
    '''Create an events file. Init, Final states for now. '''
    import numpy as np
    with open('./results/trajectory.txt','r') as myfile:
        head = [next(myfile) for x in xrange(3)]
    with open('./results/event.dat','w') as events:
        for line in ['#Events file\n','state'+15*' ' + head[1],'-'+19*' '+head[2]]:
            events.write(line)
        data = np.loadtxt('./results/trajectory.txt',skiprows = 3,unpack=False,ndmin=2)
        # print data.shape
        # np.savetxt(events,data[(0,-1),:],fmt='%-19.4f')
        for state, row in zip(['Init','Final'],data[(0,-1),:]):
            events.write('{0:6}'.format(state))
            for val in row:
                events.write('{0:^19.4f}'.format(val))
            events.write('\n')
    



