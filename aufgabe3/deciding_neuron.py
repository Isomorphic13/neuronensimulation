from typing import Callable

import numpy as np
from aufgabe2.ode_solver_euler import euler_method
from aufgabe3.receptor_neuron import ReceptorNeuron


class DecidingNeuron:
    def __init__(self, weights : list[float], list_of_neurons : list[ReceptorNeuron]):
        self.weight0 = weights[0]
        self.weight1 = weights[1]
        self.weight2 = weights[2]
        self.weight3 = weights[3]
        self.list_of_neurons = list_of_neurons

        self.initial_state = np.array([-65, 0.6, 0.5, 0.4])
        self.tuple_of_constants = (1, 36, 120, 0.3, -77, 50, -54.387)

    def get_voltage(self, set_of_pixels : np.ndarray) -> np.ndarray:
        outer_current = 0.08 * (self.weight0 * self.list_of_neurons[0].get_voltage(set_of_pixels) + self.weight1 * self.list_of_neurons[1].get_voltage(set_of_pixels) +
                                self.weight2 * self.list_of_neurons[2].get_voltage(set_of_pixels) + self.weight3 * self.list_of_neurons[3].get_voltage(set_of_pixels))
        #print(outer_current.min(), outer_current.max(), outer_current.mean())

        def array_to_callable(I_array: np.ndarray, t_a: float, dt: float) -> Callable[[float], float]:
            '''
            Wraps a precomputed current array into a callable I(t), assuming t is sampled
            on the same grid as t_points = np.arange(t_a, t_b, dt).
            '''

            def I(t):
                index = round((t - t_a) / dt)
                return I_array[index]

            return I
        I_func = array_to_callable(outer_current, 0, 0.01)

        state = euler_method(x_0=self.initial_state, t_a=0, t_b=50, dt=0.01, I=I_func,
                             tuple_of_constants=self.tuple_of_constants)
        voltage = state[0:, 0]
        return voltage

