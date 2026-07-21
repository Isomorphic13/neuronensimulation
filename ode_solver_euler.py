from math import exp
import numpy as np


def alpha_n (V):
    return -0.01 * (55 + V) / (exp(-(55 + U)/10) - 1)

def alpha_m (V):
    return -0.1 * (40 + V) / (exp(-(40 + U)/10) - 1)

def alpha_h (V):
    return 0.07 * exp(-(65 + V) / 20)

def beta_n (V):
    return 0.125 * exp(-(65 + V) / 80)

def beta_m (V):
    return 4 * exp(-(65 + V) / 18)

def beta_h (V):
    return 1 / (exp(-(35 + V) / 10) + 1)


def diff_eq_n(V,n):
    return alpha_n(V) * (1 - n) - beta_n(V) * n

def diff_eq_m(V,m):
    return alpha_m(V) * (1 - m) - beta_m(V) * m

def diff_eq_h(V,h):
    return alpha_h(V) * (1 - h) - beta_h(V) * h


def euler_methode(x_0 : np.ndarray, t_a : float, t_b : float, dt : float, voltage_0 : float, I, tuple_of_constants : tuple[float]) -> np.ndarray:
    '''
    The functiom calculates numerical values for voltage V in HHM for a single cell under some current I. The function uses the Euler methode for calculation
    :param x_0: initial values for the gating variables n,m,h
    :param t_a: initial time
    :param t_b: end time
    :param dt: time step
    :param voltage_0: inital voltage value
    :param tuple_of_constants: tuple that contains all necessary for the HHM constants
    :return: numerical values of voltage inside a single cell in time period [t_a, t_b] in form of numpy array
    '''

    #A helping fucntion returns vector of new values for the gating variables n,m,h
    def f(x : np.ndarray, voltage: float) -> np.ndarray:
        return np.array([diff_eq_n(voltage, x[0]) , diff_eq_m(voltage, x[1]), diff_eq_h(voltage, x[2])])

    #unpacking all constants
    C = tuple_of_constants[0]
    G_K = tuple_of_constants[1]
    G_NA = tuple_of_constants[2]
    G_L = tuple_of_constants[3]
    V_K = tuple_of_constants[4]
    V_NA = tuple_of_constants[5]
    V_L = tuple_of_constants[6]

    #A helping function representing the differential equation dV/dt = f_V(n(V), m(V), h(V), V(t), t) for voltage
    def f_V(x : np.ndarray, voltage ) -> float:
        n = x[0]
        m = x[1]
        h = x[2]
        return (I - G_K * (n**4) * (voltage - V_K) - G_NA * (m ** 3) * h * (voltage - V_NA) - G_L * (voltage - V_L)) / C


    #The euler procces itself.
    t_points = np.arange(t_a, t_b, dt)
    voltage_points = []
    v = voltage_0
    x = x_0
    for t in t_points:
        voltage_points.append(v)
        x, v = x + dt * f(x, v), v + dt * f_V(x, v)

    return np.array(voltage_points) #rows represent tulpe of values n,m,h at some time t_0, whereas columns represent values of a single function in the whole time interval








