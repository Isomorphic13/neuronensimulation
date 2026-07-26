
import numpy as np
from traitlets import Callable
import aufgabe2.equations as eq
from scipy.interpolate import interp1d
import scipy.integrate as spi

"""
The module implements a single neuron in the Hodgkin-Huxley model (HHM). 
"""


class Neuron:
    """
    The class represents a single neuron in the HHM model. Its instance takes current from the board input as well as from other neurons and is used to calculate the voltage inside a single neuron.
    """

    def __init__(self):
        self.initial_state = np.array([-65,0.6, 0.2, 0.75])
        self.time_array = np.arange(0, 50, 0.01)
        self.tuple_of_constants = (1, 36, 120, 0.3, -77, 50, -54.387)

    def set_time_array(self, time_array: np.ndarray):
        self.time_array = time_array

    def set_initial_state(self, initial_state: list[float]):
        self.initial_state = initial_state

    def get_voltage(self, current_function : np.ndarray) -> np.ndarray:
        """

        :param current_function: ndarray, which contains the total current that applied to the neuron.
        :return:
        """

        # bringing the input current in readable for scipy.odeint form.
        interp_current = interp1d(self.time_array, current_function, kind='linear', fill_value='extrapolate')

        #defining the system of ODEs using aufgabe2.equations, where the equations are stored in callable form.
        def _hh_system(initial_state, t, interp_current, tuple_of_constants: tuple[float]):
            # Evaluate the current at the current time t

            current_t = interp_current(t)
            return eq.equations_vector(initial_state, current_t, tuple_of_constants)

        #calculation the voltage values over time period self.time_array
        solution = spi.odeint(
            _hh_system,
            self.initial_state,
            self.time_array,
            args=(interp_current, self.tuple_of_constants)
        )

        #the first of the solution contains the values of the membrane voltage.
        return solution[:, 0]

