from numpy.linalg import norm

global inputLog

#Input log methods
def inputLogInit():
    from collections import OrderedDict
    global inputLog
    inputLog = {}
    inputLog['values'] = OrderedDict()
    inputLog['units'] = OrderedDict()
    return None
    
def addEntry(name, value, unit):
    global inputLog
    #Add an entry to the inputLog that gets written at the start of the sim.
    inputLog['values'][name.split(']')[0]] = value
    if unit is None or 'dimensionless' in unit:
        inputLog['units'][name.split(']')[0]] = '-'
    else:
        inputLog['units'][name.split(']')[0]] = unit
    
    return None
    
def writeInputLog():
    
    global inputLog
    spacing = 40
    file = open('./results/inputs.dat','w') # Create summary file to write to
    file.write('#Simulation inputs: '+ str(len(inputLog.keys())) +'\n')
    file.write('Input name'+ ' '*(spacing-10)) #10 is len('Input name')
  
    
    for key in inputLog['values']:
        file.write(key+ ' '*(spacing-len(key)))
    file.write('\n')
    file.write('Value'+ ' '*(spacing-5))
    for key in inputLog['values']:
        file.write(str(inputLog['values'][key]) + ' '*(spacing-len(str(inputLog['values'][key]))))
    file.write('\n')
    file.write('Units'+ ' '*(spacing-5))        
    for key in inputLog['values']:
        file.write(str(inputLog['units'][key]) + ' '*(spacing-len(str(inputLog['units'][key]))))
    file.close()
        
        
    return None

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