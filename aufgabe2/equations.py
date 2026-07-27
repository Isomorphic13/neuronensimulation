from numba import njit
import numpy as np
from numba import njit


"""
The module contains differential equations from the Hodgkin-Huxley model that are used for later numerical calculations 
"""
@njit
def alpha_n (voltage):
    return -0.01 * (55 + voltage) / (np.exp(-(55 + voltage) / 10) - 1)
@njit
def alpha_m (voltage):
    return -0.1 * (40 + voltage) / (np.exp(-(40 + voltage) / 10) - 1)
@njit
def alpha_h (voltage):
    return 0.07 * np.exp(-(65 + voltage) / 20)
@njit
def beta_n (voltage):
    return 0.125 *np.exp(-(65 + voltage) / 80)
@njit
def beta_m (voltage):
    return 4 * np.exp(-(65 + voltage) / 18)
@njit
def beta_h (voltage):
    return 1 / (np.exp(-(35 + voltage) / 10) + 1)

@njit
def diff_eq_n(voltage,n):
    return alpha_n(voltage) * (1 - n) - beta_n(voltage) * n
@njit
def diff_eq_m(voltage,m):
    return alpha_m(voltage) * (1 - m) - beta_m(voltage) * m
@njit
def diff_eq_h(voltage,h):
    return alpha_h(voltage) * (1 - h) - beta_h(voltage) * h




def equations_vector(x_0 : np.ndarray, I, tuple_of_constants : tuple[float] ) -> np.ndarray:
    """
    This function calculates the value differential equations dy/dx = f(y,x) for the Hodgkin-Huxley model for some time t. y is the state vector for the system.
    :param x_0: initial conditions for the system. The structure is given in the form [V, n, m, h]. Where V is voltage and n, m , h are the gating variables.
    :param I: outer current applied to the cell.
    :param tuple_of_constants: set of constant, which are used for the differential equation dV/dt =
    :return: state of the system as [V, n, m, h].
    """
    C = tuple_of_constants[0]
    G_K = tuple_of_constants[1]
    G_NA = tuple_of_constants[2]
    G_L = tuple_of_constants[3]
    V_K = tuple_of_constants[4]
    V_NA = tuple_of_constants[5]
    V_L = tuple_of_constants[6]

    voltage = x_0[0]
    n, m, h = x_0[1], x_0[2], x_0[3]

    dndt = diff_eq_n(voltage=voltage, n=n)
    dmdt = diff_eq_m(voltage=voltage, m=m)
    dhdt = diff_eq_h(voltage=voltage, h=h)

    dvdt = (I
          - G_K * (n ** 4) * (voltage - V_K)
          - G_NA * (m ** 3) * h * (voltage - V_NA)
          - G_L * (voltage - V_L)) / C

    return np.array([dvdt, dndt, dmdt, dhdt])


@njit
def equations_matrix(x_0 : np.ndarray, I, tuple_of_constants : tuple[float] ) -> np.ndarray:
    C = tuple_of_constants[0]
    G_K = tuple_of_constants[1]
    G_NA = tuple_of_constants[2]
    G_L = tuple_of_constants[3]
    V_K = tuple_of_constants[4]
    V_NA = tuple_of_constants[5]
    V_L = tuple_of_constants[6]

    voltage = x_0[0, 0:]
    n, m, h = x_0[1, 0:], x_0[2, 0:], x_0[3, 0:]

    dndt = diff_eq_n(voltage=voltage, n=n)
    dmdt = diff_eq_m(voltage=voltage, m=m)
    dhdt = diff_eq_h(voltage=voltage, h=h)

    dvdt = (I
            - G_K * (n ** 4) * (voltage - V_K)
            - G_NA * (m ** 3) * h * (voltage - V_NA)
            - G_L * (voltage - V_L)) / C

    result = np.empty_like(x_0)

    result[0, :] = dvdt
    result[1, :] = dndt
    result[2, :] = dmdt
    result[3, :] = dhdt

    return result
