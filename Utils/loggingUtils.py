

def createResultsDir():
    import os
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

def loggingInit():
    # from perturbUtils import inputLogInit
    # inputLogInit()
    createResultsDir()

    return

def loggingFinish(states,units,data,summary):
    import numpy as np
    from dataUtils import writeEvents, writeSummary
    # from perturbUtils import writeInputLog
    # writeInputLog()
    if summary is not None:
        writeSummary(summary)
    fmt = '%-19.4f'
    with file('./results/trajectory.txt','w') as result:
        
        result.write('#Trajectory Data\n')
        for state in states:
            result.write("{0:20}".format(state)) 
        result.write('\n')
        for unit in units:
            result.write("{0:20}".format(unit)) 
        result.write('\n')
        np.savetxt(result, data, fmt = fmt)
    
    writeEvents()