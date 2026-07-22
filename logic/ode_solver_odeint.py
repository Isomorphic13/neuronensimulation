import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- Rate functions (alpha/beta) ---
def alpha_n(V): return -0.01 * (55 + V) / (np.exp(-(55 + V) / 10) - 1)
def alpha_m(V): return -0.1 * (40 + V) / (np.exp(-(40 + V) / 10) - 1)
def alpha_h(V): return 0.07 * np.exp(-(65 + V) / 20)

def beta_n(V):  return 0.125 * np.exp(-(65 + V) / 80)
def beta_m(V):  return 4 * np.exp(-(65 + V) / 18)
def beta_h(V):  return 1 / (np.exp(-(35 + V) / 10) + 1)


# --- The combined ODE system: state = [V, n, m, h] ---
def hh_system(state, t, I, tuple_of_constants : tuple[float]):
    C, G_K, G_NA, G_L, V_K, V_NA, V_L = tuple_of_constants

    V, n, m, h = state

    dVdt = (I - G_K * n**4 * (V - V_K)
              - G_NA * m**3 * h * (V - V_NA)
              - G_L * (V - V_L)) / C

    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h

    return [dVdt, dndt, dmdt, dhdt]






