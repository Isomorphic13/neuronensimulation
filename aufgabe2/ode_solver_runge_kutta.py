from typing import Callable

from aufgabe2 import equations as eq
import numpy as np

def runge_kutta_method(x_0 : np.ndarray, t_a : float, t_b : float, dt : float, I : Callable[[float], float], tuple_of_constants : tuple[float]) -> np.ndarray:
    '''
        The functiom calculates numerical values for voltage V in HHM for a single cell under some current I. The function uses the Euler methode for calculation
        :param x_0: initial values for the gating variables n,m,h
        :param t_a: initial time
        :param t_b: end time
        :param dt: time step
        :param voltage_0: inital voltage value
        :param I: current at the time
        :param tuple_of_constants: tuple that contains all necessary for the HHM constants
        :return: numerical values of voltage inside a single cell in time period [t_a, t_b] in form of numpy array
        '''

    t_points = np.arange(t_a, t_b, dt)
    values_of_functions = []

    x = x_0.copy()

    for t in t_points:
        values_of_functions.append(x)
        I_current = I(t)
        k1 = dt * eq.equations_vector(x=x, I=I_current, tuple_of_constants=tuple_of_constants)
        k2 = dt * eq.equations_vector(x=x + k1 / 2, I=I_current, tuple_of_constants=tuple_of_constants)
        k3 = dt * eq.equations_vector(x=x + k2 / 2, I=I_current, tuple_of_constants=tuple_of_constants)
        k4 = dt * eq.equations_vector(x=x + k3, I=I_current, tuple_of_constants=tuple_of_constants)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return np.array(values_of_functions)