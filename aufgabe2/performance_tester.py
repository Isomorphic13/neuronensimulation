import numbers
import time

from aufgabe2.ode_solver_euler import euler_method
from aufgabe2.ode_solver_runge_kutta import runge_kutta_method
from aufgabe2.ode_solver_odeint import solve_with_odeint
import numpy as np
import matplotlib.pyplot as plt

number_of_tests = 10

def test_euler_runtimes(x_0 : np.ndarray, t_a: float, t_b :float , time_steps : np.ndarray, I_0 : float, tuple_of_constants : tuple[float]) -> np.ndarray:
    '''
    :param x_0: initial condition
    :param t_a: beginning time of a single simulation
    :param t_b: ending time of a single simulation
    :param time_steps: time steps dt for numerical solver
    :param I_0: initial current
    :param tuple_of_constants: constants for the Hodgkin - Huxley model
    :return: runtime for different time steps
    '''

    all_runtimes = np.zeros_like(time_steps) #array to save all runtimes value, which will be divided by the number of tests
    single_runtimes = np.empty_like(time_steps)

    I = lambda t: I_0
    for n in range(number_of_tests):
        for i in range(time_steps.shape[0]): #time_steps contains values of dt, which will be passed to the ODE-solver
            try:
                start = time.perf_counter()
                sol = euler_method(x_0, t_a, t_b, time_steps[i], I, tuple_of_constants)
                end = time.perf_counter()
                single_runtimes[i] = end - start #runtime value

            except (OverflowError, FloatingPointError): #since the method will become numerically  instable beginning from some time step, we should handle this
                single_runtimes[i] = np.nan

        all_runtimes = all_runtimes + single_runtimes

    return all_runtimes / number_of_tests #getting the mean value of runtime

def test_runge_kutta_runtimes(x_0 : np.ndarray, t_a: float, t_b :float , time_steps : np.ndarray, I_0 : float, tuple_of_constants : tuple[float]) -> np.ndarray:
    all_runtimes = np.zeros_like(time_steps)
    single_runtimes = np.empty_like(time_steps)

    I = lambda t: I_0
    for n in range(number_of_tests):
        for i in range(time_steps.shape[0]):
            try:
                start = time.perf_counter()
                sol = runge_kutta_method(x_0, t_a, t_b, time_steps[i], I, tuple_of_constants)
                end = time.perf_counter()
                single_runtimes[i] = end - start

            except (OverflowError, FloatingPointError):
                single_runtimes[i] = np.nan

        all_runtimes = all_runtimes + single_runtimes

    return all_runtimes / number_of_tests

def test_odeint(x_0 : np.ndarray, t_a: float, t_b :float , time_steps : np.ndarray, I_0 : float, tuple_of_constants : tuple[float]) -> np.ndarray:
    all_runtimes = np.zeros_like(time_steps)
    single_runtimes = np.empty_like(time_steps)

    for n in range(number_of_tests):
        for i in range(time_steps.shape[0]):
            time_points = np.arange(t_a, t_b, time_steps[i])
            try:
                start = time.perf_counter()
                sol = solve_with_odeint(time_points, x_0, I_0, tuple_of_constants)
                end = time.perf_counter()
                single_runtimes[i] = end - start

            except (OverflowError, FloatingPointError):
                single_runtimes[i] = np.nan

        all_runtimes = all_runtimes + single_runtimes

    return all_runtimes / number_of_tests

