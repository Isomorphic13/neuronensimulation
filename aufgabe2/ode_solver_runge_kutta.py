from typing import Callable

from aufgabe2 import equations as eq
import numpy as np
from numba import njit

"""
The module implements the Runge-Kutta method for numerical solving of system ordinary differential equations from Hodgkin-Huxley model (HHM).
"""


def runge_kutta_method(x_0 : np.ndarray, t_a : float, t_b : float, dt : float, I : Callable[[float], float], tuple_of_constants : tuple[float]) -> np.ndarray:
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
        k1 = dt * eq.equations_vector(x_0=x, I=I_current, tuple_of_constants=tuple_of_constants)
        k2 = dt * eq.equations_vector(x_0=x + k1 / 2, I=I_current, tuple_of_constants=tuple_of_constants)
        k3 = dt * eq.equations_vector(x_0=x + k2 / 2, I=I_current, tuple_of_constants=tuple_of_constants)
        k4 = dt * eq.equations_vector(x_0=x + k3, I=I_current, tuple_of_constants=tuple_of_constants)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return np.array(values_of_functions)


@njit
def runge_kutta_method_matrix(x_0 : np.ndarray, time_array : np.ndarray, dt : float, eigen_current :np.ndarray, weights_matrix: np.ndarray, tuple_of_constants : tuple) -> np.ndarray:
    '''
    :param x_0: inital state matrix. Each row corresponds to the inital state of a single cell.
    :param time_array: time interval
    :param dt: time step
    :param eigen_current: the current that each cell becomes without considering the currents from other cells.
    :param weights_matrix: matrix that is used to find the current state of the network according to i = Wv. The matrix contains information about the topology of a network.
    :param tuple_of_constants: tuple that contains all necessary for the HHM constants.
    :return: tensor of neuronal network over time. It has structure [time t, [state of all neurons in time t]]. [state of all neurons in time t] = [[state_of_neuron1], [state_of_neuron2], ...]
    '''
    values_of_functions = np.empty(
        (len(time_array), x_0.shape[0], x_0.shape[1]),
        dtype=np.float64
    )

    x = x_0.copy()

    for i in range(len(time_array)):

        values_of_functions[i, :, :] = x

        # Calculate the current vector
        outer_current = weights_matrix @ x[0, :]
        total_current = outer_current + eigen_current

        k1 = dt * eq.equations_matrix(
            x_0=x,
            I=total_current,
            tuple_of_constants=tuple_of_constants
        )
        k2 = dt * eq.equations_matrix(
            x_0=x + k1 / 2.0,
            I=total_current,
            tuple_of_constants=tuple_of_constants
        )
        k3 = dt * eq.equations_matrix(
            x_0=x + k2 / 2.0,
            I=total_current,
            tuple_of_constants=tuple_of_constants
        )
        k4 = dt * eq.equations_matrix(
            x_0=x + k3,
            I=total_current,
            tuple_of_constants=tuple_of_constants
        )
        x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return values_of_functions
