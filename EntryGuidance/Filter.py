''' Defines filters for estimation '''

from numpy import sin, cos, arctan2
from numpy.linalg import norm
import numpy as np


def FadingMemory(currentValue, measuredValue, gain):
    return (1-gain)*(measuredValue-currentValue)
    