import aufgabe2.equations as eq
import numpy as np
import scipy.integrate as spi


def _hh_system(state : np.ndarray, t, I : float, tuple_of_constants : tuple[float]):
    return eq.equations_vector(state, I, tuple_of_constants)


def solve_with_odeint(time_array : np.ndarray, state_0 : np.ndarray, I: float, tuple_of_constants : tuple[float] ):
    solution = spi.odeint(_hh_system, state_0, time_array, args=(I, tuple_of_constants))

    V_t = solution[:, 0]
    return V_t