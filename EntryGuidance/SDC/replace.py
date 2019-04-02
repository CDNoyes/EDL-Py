import numpy as np 


def replace(x, replace_value):
    """ A useful method for SDC factorizations. """
    if np.isfinite(x):
        return x
    else:
        return replace_value
