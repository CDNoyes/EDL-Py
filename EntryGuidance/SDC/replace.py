import numpy as np 
import warnings 

def safe_divide(num, den, replace_value):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return replace(num/den, replace_value)

def replace(x, replace_value):
    """ A useful method for SDC factorizations. """
    if np.isfinite(x):
        return x
    else:
        return replace_value
