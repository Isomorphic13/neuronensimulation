from . import equations as eq
import numpy as np

def runge_kutta_method(x_0 : np.ndarray, t_a : float, t_b : float, dt : float, voltage_0 : float, I, tuple_of_constants : tuple[float]) -> np.ndarray:
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

        k1_x = dt * eq.f_gating(x, v)
        k1_v = dt * eq.f_V(x=x, v = v, I=I, tuple_of_constants=tuple_of_constants)

        k2_x = dt * eq.f_gating(x + k1_x / 2, v + k1_v / 2)
        k2_v = dt * eq.f_V(x=x + k1_x / 2, v = v, I=I, tuple_of_constants=tuple_of_constants)

        k3_x = dt * eq.f_gating(x + k2_x / 2, v + k2_v / 2)
        k3_v = dt * eq.f_V(x=x + k2_x / 2, v = v, I=I, tuple_of_constants=tuple_of_constants)

        k4_x = dt * eq.f_gating(x + k3_x, v + k3_v)
        k4_v = dt * eq.f_V(x=x + k3_x, v = v, I=I, tuple_of_constants=tuple_of_constants)

        x = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        v = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return np.array(voltage_points)