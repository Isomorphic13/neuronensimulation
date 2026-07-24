import numpy as np
from aufgabe2.ode_solver_euler import euler_method

class Neuron:
    def __init__(self, index: list[int]):
        self.index = index
        self.initial_state = np.array([-65,0.6, 0.5, 0.4])
        self.tuple_of_constants = (1, 36, 120, 0.3, -77, 50, -54.387)

    def get_pixel_value(self):
