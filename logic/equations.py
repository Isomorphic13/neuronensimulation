from math import exp
import numpy as np


def alpha_n (V):
    return -0.01 * (55 + V) / (exp(-(55 + V)/10) - 1)

def alpha_m (V):
    return -0.1 * (40 + V) / (exp(-(40 + V)/10) - 1)

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

#Function returns vector of new values for the gating variables n,m,h
def f_gating(x : np.ndarray, voltage: float) -> np.ndarray:
    return np.array([diff_eq_n(voltage, x[0]) , diff_eq_m(voltage, x[1]), diff_eq_h(voltage, x[2])])

#Function representing the differential equation dV/dt = f_V(n(V), m(V), h(V), V(t), t) for voltage
def f_V(x : np.ndarray, v, I, tuple_of_constants : tuple[float] ) -> float:
    C = tuple_of_constants[0]
    G_K = tuple_of_constants[1]
    G_NA = tuple_of_constants[2]
    G_L = tuple_of_constants[3]
    V_K = tuple_of_constants[4]
    V_NA = tuple_of_constants[5]
    V_L = tuple_of_constants[6]

    n = x[0]
    m = x[1]
    h = x[2]
    return (I - G_K * (n**4) * (v - V_K) - G_NA * (m ** 3) * h * (v - V_NA) - G_L * (v - V_L)) / C