from . import equations as eq
import numpy as np

def euler_method(x_0 : np.ndarray, t_a : float, t_b : float, dt : float, voltage_0 : float, I, tuple_of_constants : tuple[float]) -> np.ndarray:
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
    voltage_points = []
    v = voltage_0
    x = x_0
    for t in t_points:
        voltage_points.append(v)
        x, v = x + dt * eq.f_gating(x, v), v + dt * eq.f_V(x=x, v=v, I=I, tuple_of_constants=tuple_of_constants)

    return np.array(voltage_points)








