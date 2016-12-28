''' Monte Carlo Filtering module'''

import numpy as np

def mcfilter(B, NB, input_names = None, threshold=0):
    ''' Performs monte carlo filtering on data that has been split into behavioral and non-behavioral sets. '''
    
    from scipy.stats import ks_2samp as kstest2
    
    # Input checking
    if input_names is None or (len(input_names) != B.shape[0]):
        # print("Using generic names in MCF.filter")
        input_names = ["Input{}:".format(i) for i in range(B.shape[0])] # Generic names
    else:
        input_names = [inp + ':' for inp in input_names]
        
    max_len = np.max([len(name) for name in input_names])    
    max_str = '<{}'.format(max_len)
    ks_values = []
    p_values = []
    inputs = []    
    
    for i in range(B.shape[0]):
        ks, p = kstest2(B[i,:], NB[i,:])
        if ks >= threshold:
            inputs.append(i)
            ks_values.append(ks)
            p_values.append(p)
    
    # Sort and display        
    ks_values_sorted, p_values_sorted, inputs_sorted = zip(*sorted(zip(ks_values,p_values,inputs),reverse=True))
    print "\n{}   KS      P".format('Input'.ljust(max_len))
    print "-"*(max_len+15)
    for i in inputs_sorted:
        print "{name: {fill}}  {ks:.2f}   {p:.3f}".format(name=input_names[i], ks=ks_values[i], p=p_values[i], fill=max_str)
    print '\n'    
        
def mcsplit(inputs, outputs, criteria):
    ''' 
        Generates behavioral/non-behavioral input arrays for use in MCF.
        
        Inputs:
            inputs - A 2-D array
            outputs - A list/tuple of 2-D arrays
            criteria - A function operating on a 2-D array returning a boolean
            
        Outputs:
            B - behavioral partition of the inputs
            NB - non-behavioral partition of the inputs
            
    '''
    
    b = np.array([criteria(output) for output in outputs],dtype=bool)
    nb = np.array([not bb for bb in b], dtype=bool)
    B = inputs[:,b]
    NB = inputs[:,nb]
    return B, NB
    
if __name__ == "__main__":

    inputs = 2*(np.random.rand(3,5000)-0.5)
    outputs = [10*inp[1]**2 + 2*inp[2] for inp in inputs.T] # Strong dependence on inp[1], mild on inp[2], no dependence on inp[0]
    def my_fun(output):
        return output>3
        
    data = mcsplit(inputs,outputs,my_fun)
    mcfilter(*data, input_names=['Really Long Name Is Long','Strong','Mild']) 