from typing import Callable

from aufgabe2 import equations as eq
import numpy as np

"""
The module implements the Euler method for numerical solving system ordinary differential equations from Hodgkin-Huxley model (HHM).
"""

def euler_method(x_0 : np.ndarray, t_a : float, t_b : float, dt : float, I : Callable[[float], float], tuple_of_constants : tuple[float]) -> np.ndarray:
    '''
    The functiom calculates numerical values for voltage V in HHM for a single cell under some current I. The function uses the Euler methode for calculation.
    :param x_0: initial values for voltage V and gating variables n,m,h.
    :param t_a: initial time.
    :param t_b: end time.
    :param dt: time step.
    :param I: current function.
    :param tuple_of_constants: tuple that contains all necessary for the HHM constants.
    :return: numerical values of voltage V and gating variables n,m,h inside a single cell in time period [t_a, t_b] in form of numpy array. The returns the result as a numpy array of shape t x 4, where t is the time array.
    '''

    t_points = np.arange(t_a, t_b, dt)
    values_of_functions = []
    x = x_0.copy()
    for t in t_points:
        values_of_functions.append(x)
        I_current = I(t)
        x = x + dt * eq.equations_vector(x_0=x, I=I_current, tuple_of_constants=tuple_of_constants)
    return np.array(values_of_functions)








