from . import equations as eq
import numpy as np

def runge_kutta_method(x_0 : np.ndarray, t_a : float, t_b : float, dt : float, voltage_0 : float, I, tuple_of_constants : tuple[float]) -> np.ndarray:
    t_points = np.arange(t_a, t_b, dt)
    voltage_points = []
    v = voltage_0
    x = x_0

    for t in t_points:
        voltage_points.append(v)

        k1_x = dt * eq.f_gating(x, v)
        k1_v = dt * eq.f_V(x=x, v=v, I=I, tuple_of_constants=tuple_of_constants)

        k2_x = dt * eq.f_gating(x + k1_x / 2, v + k1_v / 2)
        k2_v = dt * eq.f_V(x=x + k1_x / 2, v=v + k1_v / 2, I=I, tuple_of_constants=tuple_of_constants)

        k3_x = dt * eq.f_gating(x + k2_x / 2, v + k2_v / 2)
        k3_v = dt * eq.f_V(x=x + k2_x / 2, v=v + k2_v / 2, I=I, tuple_of_constants=tuple_of_constants)

        k4_x = dt * eq.f_gating(x + k3_x, v + k3_v)
        k4_v = dt * eq.f_V(x=x + k3_x, v=v + k3_v, I=I, tuple_of_constants=tuple_of_constants)

        x = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        v = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return np.array(voltage_points)  # rows represent collection of values n,m,h at some time t_0, whereas columns represent values of a single function in the whole time interval