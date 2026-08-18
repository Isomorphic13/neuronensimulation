from numba import njit
import numpy as np
from numba import njit
from math import exp


"""
The module contains differential equations from the Hodgkin-Huxley model that are used for later numerical calculations 
"""
'''
These are equations for coefficients alpha(V), beta(V) as well as differential equations for the gating variables f_n, f_m , f_h.
There is two form of this equations, for scalers and 1D - ndarrays. They are used depending on whether you calculate state of a single neuron or the whole neuronal network.
For the acceleration purpose 'numba' library was used for the neuronal network. 
'''

def alpha_n_scalar(voltage):
    x = -(55 + voltage) / 10
    if abs(x) > 1e-6:
        return 0.1 * x / (exp(x) - 1)
    else:   #Taylor expansion as voltage approaches v_0 = -55
        return 0.1 - x/20 + x**2 / 120

def alpha_m_scalar(voltage):
    x = -(40 + voltage) / 10
    if abs(x) > 1e-6:
        return x / (exp(x) - 1)
    else: #Taylor expansion as voltage approaches v_0 = -40
        return 1 - x / 2 + x ** 2 / 12

def alpha_h_scalar(voltage):
    return 0.07 * np.exp(-(65 + voltage) / 20)

def beta_n_scalar(voltage):
    return 0.125 * np.exp(-(65 + voltage) / 80)

def beta_m_scalar(voltage):
    return 4 * np.exp(-(65 + voltage) / 18)

def beta_h_scalar(voltage):
    return 1 / (
        np.exp(-(35 + voltage) / 10) + 1
    )

def diff_eq_n_scalar(voltage, n):
    return (
        alpha_n_scalar(voltage) * (1 - n)
        - beta_n_scalar(voltage) * n
    )

def diff_eq_m_scalar(voltage, m):
    return (
        alpha_m_scalar(voltage) * (1 - m)
        - beta_m_scalar(voltage) * m
    )

def diff_eq_h_scalar(voltage, h):
    return (
        alpha_h_scalar(voltage) * (1 - h)
        - beta_h_scalar(voltage) * h
    )

#vector version of the equations above
@njit
def alpha_n_vector(voltage):
    result = np.empty_like(voltage)
    for i in range(voltage.shape[0]):
        x  = -(55 + voltage[i]) / 10
        if abs(x) > 1e-6:
            result[i] = 0.1 * x /(np.exp(x) - 1)
        else:
            result[i] = 0.1 - x / 20 + x ** 2 / 120 #Taylor expansion as voltage approaches v_0 = -55 in a single cell
    return result

@njit
def alpha_m_vector(voltage):
    result = np.empty_like(voltage)
    for i in range(voltage.shape[0]):
        x = -(40 + voltage[i]) / 10
        if abs(x) > 1e-6:
            result[i] = x / (np.exp(x) - 1)
        else:
            result[i] = 1 - x / 2 + x ** 2 / 12 #Taylor expansion as voltage approaches v_0 = -55 in a single cell
    return result

@njit
def alpha_h_vector (voltage):
    return 0.07 * np.exp(-(65 + voltage) / 20)
@njit
def beta_n_vector (voltage):
    return 0.125 *np.exp(-(65 + voltage) / 80)
@njit
def beta_m_vector (voltage):
    return 4 * np.exp(-(65 + voltage) / 18)
@njit
def beta_h_vector (voltage):
    return 1 / (np.exp(-(35 + voltage) / 10) + 1)

@njit
def diff_eq_n_vector(voltage, n):
    return (
        alpha_n_vector(voltage) * (1 - n)
        - beta_n_vector(voltage) * n
    )

@njit
def diff_eq_m_vector(voltage, m):
    return (
        alpha_m_vector(voltage) * (1 - m)
        - beta_m_vector(voltage) * m
    )

@njit
def diff_eq_h_vector(voltage, h):
    return (
        alpha_h_vector(voltage) * (1 - h)
        - beta_h_vector(voltage) * h
    )


def equations_vector(x_0 : np.ndarray, I, tuple_of_constants : tuple[float] ) -> np.ndarray:
    """
    This function calculates the value differential equations dy/dx = f(y,x) for the Hodgkin-Huxley model for some time t. It returns the state vector [V, n , m, h] for the system.
    The function is called for to compute the next value of the state in Euler or Runge-Kutta methods.
    :param x_0: initial conditions for the system. The structure is given in the form [V, n, m, h]. Where V is voltage and n, m , h are the gating variables.
    :param I: outer current applied to the cell.
    :param tuple_of_constants: set of constant, which are used for the differential equation dV/dt
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

    dndt = diff_eq_n_scalar(voltage=voltage, n=n)
    dmdt = diff_eq_m_scalar(voltage=voltage, m=m)
    dhdt = diff_eq_h_scalar(voltage=voltage, h=h)

    dvdt = (I
          - G_K * (n ** 4) * (voltage - V_K)
          - G_NA * (m ** 3) * h * (voltage - V_NA)
          - G_L * (voltage - V_L)) / C

    return np.array([dvdt, dndt, dmdt, dhdt]) #return the numerical values of ODE of the system for time some time point

@njit
def equations_matrix(x_0 : np.ndarray, I, tuple_of_constants : tuple[float] ) -> np.ndarray:
    '''

    :param x_0: initial state matrix
    :param I: current vector, which elements correspond to the currents on single cell.
    :param tuple_of_constants: set of constant, which are used for the differential equation dV/dt.
    :return: matrix of the state. Each row corresponds to the change of the state of single cell in time t.
    '''
    C = tuple_of_constants[0]
    G_K = tuple_of_constants[1]
    G_NA = tuple_of_constants[2]
    G_L = tuple_of_constants[3]
    V_K = tuple_of_constants[4]
    V_NA = tuple_of_constants[5]
    V_L = tuple_of_constants[6]

    voltage = x_0[0, 0:]
    n, m, h = x_0[1, 0:], x_0[2, 0:], x_0[3, 0:]

    dndt = diff_eq_n_vector(voltage=voltage, n=n)
    dmdt = diff_eq_m_vector(voltage=voltage, m=m)
    dhdt = diff_eq_h_vector(voltage=voltage, h=h)

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
