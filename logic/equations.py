from math import exp
import numpy as np


def alpha_n (voltage):
    return -0.01 * (55 + voltage) / (exp(-(55 + voltage) / 10) - 1)

def alpha_m (voltage):
    return -0.1 * (40 + voltage) / (exp(-(40 + voltage) / 10) - 1)

def alpha_h (voltage):
    return 0.07 * exp(-(65 + voltage) / 20)

def beta_n (voltage):
    return 0.125 * exp(-(65 + voltage) / 80)

def beta_m (voltage):
    return 4 * exp(-(65 + voltage) / 18)

def beta_h (voltage):
    return 1 / (exp(-(35 + voltage) / 10) + 1)


def diff_eq_n(voltage,n):
    return alpha_n(voltage) * (1 - n) - beta_n(voltage) * n

def diff_eq_m(voltage,m):
    return alpha_m(voltage) * (1 - m) - beta_m(voltage) * m

def diff_eq_h(voltage,h):
    return alpha_h(voltage) * (1 - h) - beta_h(voltage) * h



#Function representing the differential equation dV/dt = f_V(n(V), m(V), h(V), V(t), t) for voltage and dn/dt, dm/dt, dh/dt
def equations_vector(x : np.ndarray, I, tuple_of_constants : tuple[float] ) -> np.ndarray:
    C = tuple_of_constants[0]
    G_K = tuple_of_constants[1]
    G_NA = tuple_of_constants[2]
    G_L = tuple_of_constants[3]
    V_K = tuple_of_constants[4]
    V_NA = tuple_of_constants[5]
    V_L = tuple_of_constants[6]

    voltage = x[0]
    n, m, h = x[1], x[2], x[3]

    dndt = diff_eq_n(voltage=voltage, n=n)
    dmdt = diff_eq_m(voltage=voltage, m=m)
    dhdt = diff_eq_h(voltage=voltage, h=h)

    dvdt = (I
          - G_K * (n ** 4) * (voltage - V_K)
          - G_NA * (m ** 3) * h * (voltage - V_NA)
          - G_L * (voltage - V_L)) / C

    return np.array([dvdt, dndt, dmdt, dhdt])