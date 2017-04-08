import numpy as np


from scipy.integrate import trapz
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from functools import partial
from numpy import pi
from ParametrizedPlanner import HEPBank


def controller(trigger):