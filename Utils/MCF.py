''' Monte Carlo Filtering module'''

import numpy as np
import pandas as pd 

def mcfilter(B, NB, input_names=None, ks_threshold=0, p_threshold=1e-3, plot=False, fontsize=16):
    ''' Performs monte carlo filtering on data that has been split into behavioral and non-behavioral sets.
    
    Inputs:
        B               Behavioral dataset
        NB              Non-behavioral dataset
        input_names     An optional list of names for the inputs. If None, the inputs are simply enumerated.
        ks_threshold    Lower limit on KS statistic for significance
        p_threshold     Upper limit on p-value for significance
        plot            Option to generate CDFs of the significant inputs 
        fontsize        Fontsize argument in plot 
    
    
    '''

    from scipy.stats import ks_2samp as kstest2

    # Input checking
    if not B.shape[1] or not NB.shape[1]:
        print("Cannot filter data due to 100% behavioral or non-behavioral.\n")
        return
    if input_names is None or (len(input_names) != B.shape[0]):
        input_names = ["Input{}".format(i) for i in range(B.shape[0])]  # Generic names
    # else:
    #     input_names = [inp for inp in input_names]

    max_len = np.max([len(name) for name in input_names]+[5])    
    max_str = '<{}'.format(max_len)
    ks_values = []
    p_values = []
    inputs = []
    index = []

    for i in range(B.shape[0]):
        ks, p = kstest2(B[i,:], NB[i,:])
        if ks >= ks_threshold and p <= p_threshold:
            ks_values.append(ks)
            p_values.append(p)
            inputs.append(input_names[i])
            index.append(i)

    # Sort and print
    if ks_values:
        ks_values_sorted, p_values_sorted, inputs_sorted = zip(*sorted(zip(ks_values, p_values, inputs), reverse=True))
        print("\n{}   KS      P".format('Input'.ljust(max_len)))
        print("-"*(max_len+15))
        for name, value, pvalue in zip(inputs_sorted, ks_values_sorted, p_values_sorted):
            print("{name: {fill}}  {ks:.2f}   {p:.3f}".format(name=name, ks=value, p=pvalue, fill=max_str))
        print('\n')

        # Display
        if plot:
            import matplotlib.pyplot as plt
            for i in index:
                _ecdf(B[i,:], NB[i,:], input_names[i], fontsize=fontsize)
            plt.show()
        return pd.DataFrame({"Variable" : inputs_sorted, "KS" : ks_values_sorted, "p-value" : p_values_sorted})
    else:
        print("No significant inputs found.")
        return 


def mcsplit(inputs, outputs, criteria):
    '''
        Generates behavioral/non-behavioral input arrays for use in MCF.

        Inputs:
            inputs - A 2-D array
            outputs - A list/tuple of output objects - can take any form
            criteria - A function operating on each output that returns a boolean

        Outputs:
            B - behavioral partition of the inputs
            NB - non-behavioral partition of the inputs

    '''

    b = np.array([criteria(output) for output in outputs],dtype=bool)
    nb = np.array([not bb for bb in b], dtype=bool)
    print("{} cases ({}%) are non-behavioral.".format(np.sum(nb), 100.0*np.sum(nb)/float(inputs.shape[1])))
    B = inputs[:,b]
    NB = inputs[:,nb]
    return B, NB


def _getECDF(data):
    """ Computes the x and y values for an ECDF plot. """
    x = np.sort(data)
    return x, np.arange(1, len(x)+1)/float(len(x))


def _ecdf(dataB, dataNB, name, fontsize):
    """ Plots the ECDF of two arrays. """
    import matplotlib.pyplot as plt

    x, y = _getECDF(dataB)
    xn, yn = _getECDF(dataNB)

    plt.figure()
    plt.plot(x, y, label='Behavioral')
    plt.plot(xn, yn, label='Non-Behavioral')
    plt.legend(loc='best')
    plt.xlabel(name, fontsize=fontsize)
    plt.ylabel('CDF', fontsize=fontsize)


if __name__ == "__main__":

    inputs = 2*(np.random.rand(3, 5000)-0.5)
    inputs[1] = np.random.normal(0, 1/3, 5000)
    outputs = [10*inp[1]**2 + 2*inp[2] for inp in inputs.T]  # No dependence on inp[0], Strong dependence on inp[1], mild on inp[2]

    def my_fun(output):
        return output > 2

    data = mcsplit(inputs, outputs, my_fun)
    # mcfilter(*data, input_names=['Really Long Name Is Long', 'Strong', 'Mild'], plot=True, ks_threshold=0, p_threshold=0.3)
    # df = mcfilter(*data, input_names=['X', 'Y', 'Z'], plot=True, ks_threshold=0, p_threshold=1.0)
    # print(df)
    # df.to_hdf(".\MCFN.h5", "table", mode="w")