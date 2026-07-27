from aufgabe2.ode_solver_euler import euler_method
from aufgabe2.ode_solver_runge_kutta import runge_kutta_method
import matplotlib.pyplot as plt
import numpy as np

V_0 = -65
n_0 = 0.6
m_0 = 0.5
h_0 = 0.4
x = np.array([V_0,n_0, m_0, h_0])
tuple_of_constants = (1, 36, 120, 0.3, -77, 50, -54.387)


def visualize_a(x_0, I_0, t_a, t_b, dt):
    I = lambda t: I_0
    v = euler_method(x_0, t_a, t_b, dt, I, tuple_of_constants) [0: ,0]
    tpoints = np.arange(0, 50, dt)
    plt.figure(figsize=(10, 5))
    plt.plot(tpoints,v, linewidth=2)
    plt.title(
        "Membran-Spannung über die Zeit berechnet mit dem Euler-Verfahren\n"
        + rf"$V_0 = {x_0[0]}\,$mV, $n_0 = {x_0[1]}, m_0 = {x_0[2]}, h_0 = {x_0[3]}, dt = {dt}$"
    )
    plt.xlabel("Zeit t, ms")
    plt.ylabel("Spannung V, mV")
    plt.grid(True)
    plt.show()


def visualize_b(x_0, I_0, t_a, t_b, dt):
    I = lambda t: I_0
    tpoints = np.arange(0, 50, dt)
    v = runge_kutta_method(x_0, t_a, t_b, dt, I, tuple_of_constants)[0:, 0]
    plt.figure(figsize=(10, 5))
    plt.plot(tpoints, v, linewidth=2)
    plt.title(
        "Membran-Spannung über die Zeit berechnet mit dem Runge-Kutta-Verfahren\n"
        + rf"$V_0 = {x_0[0]}\,$mV, $n_0 = {x_0[1]}, m_0 = {x_0[2]}, h_0 = {x_0[3]}, dt = {dt}$"
    )
    plt.xlabel("Zeit t, ms")
    plt.ylabel("Spannung V, mV")
    plt.grid(True)
    plt.show()

def visualize_c(x_0, t_a, t_b, dt):
    plt.figure(figsize=(10, 5))

    current_values = list(range(-5, 6))
    tpoints = np.arange(t_a, t_b, dt)

    for i in current_values:
        I_func = lambda t, val=i: val
        temp = euler_method(x_0, t_a, t_b, dt, I_func, tuple_of_constants)[0:, 0]
        plt.plot(tpoints, temp, linewidth=2, label=f"$I_0 = {i}$nA")

    plt.xlabel("Zeit $t$, ms")
    plt.ylabel("Spannung $V$, mV")
    plt.title("Membranspannung über Zeit für verschiedene Stromstärken.")
    plt.grid(True)

    plt.legend(loc="upper right", fontsize="small")

    # another plot
    plt.figure(figsize=(10, 5))

    current_values = list(range(6, 16))
    tpoints = np.arange(t_a, t_b, dt)

    for i in current_values:
        I_func = lambda t, val=i: val
        temp = euler_method(x_0, t_a, t_b, dt, I_func, tuple_of_constants)[0:, 0]
        plt.plot(tpoints, temp, linewidth=2, label=f"$I_0 = {i}$nA")

    plt.xlabel("Zeit $t$, ms")
    plt.ylabel("Spannung $V$, mV")
    plt.title("Membranspannung über Zeit für verschiedene Stromstärken.")
    plt.grid(True)

    plt.legend(loc="upper right", fontsize="small")

    plt.show()

def visualize_d(x_0, t_a, t_b, dt):
    I_0 = 4
    tpoints = np.arange(t_a, t_b, dt)
    I_func = lambda t: 50 if (10 <= t <= 11) else I_0
    I_values = np.array([I_func(t) for t in tpoints])
    s = euler_method(x_0, t_a, t_b, dt, I_func, tuple_of_constants)
    v = s[0:, 0]
    n = s[0:, 1]
    m = s[0:, 2]
    h = s[0:, 3]
    tpoints = np.arange(0, 50, dt)

    plt.figure(figsize=(10, 5))
    plt.plot(tpoints, I_values, linewidth=2)
    plt.xlabel("Zeit $t$, ms")
    plt.ylabel("Strom $I$, nA")
    plt.title(
        "Äußerer Strom $I$ über die Zeit.\n"
        + rf"$I = 50\,$nA im Intervall $t \in [10,11]$ und {I_0} überall sonst."
    )
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(tpoints, v, linewidth=2)
    plt.xlabel("Zeit $t$, ms")
    plt.ylabel("Spannung $V$, mV")
    plt.title("Membran-Spannung $V$ über die Zeit für einen nicht-konstanten Strom.")
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(tpoints, m, linewidth=2)
    plt.xlabel("Zeit $t$, ms")
    plt.ylabel("$m$")
    plt.title("Gating-Variable $m$ über die Zeit.")
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(tpoints, h, linewidth=2)
    plt.xlabel("Zeit $t$, ms")
    plt.ylabel("$h$")
    plt.title("Gating-Variable $h$ über die Zeit.")
    plt.grid(True)

    plt.show()