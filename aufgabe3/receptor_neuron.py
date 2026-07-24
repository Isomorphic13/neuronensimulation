import numpy as np
from aufgabe2.ode_solver_euler import euler_method

class ReceptorNeuron:
    def __init__(self, index: int):
        self.index = index
        self.initial_state = np.array([-65,0.6, 0.5, 0.4])
        self.tuple_of_constants = (1, 36, 120, 0.3, -77, 50, -54.387)

    def get_voltage(self, field: np.ndarray) -> np.ndarray:
        current = 15 if field[self.index] == 1 else -5
        current_func = lambda t: current
        state = euler_method(x_0 = self.initial_state, t_a = 0, t_b = 50, dt =0.01, I = current_func, tuple_of_constants = self.tuple_of_constants)
        voltage = state[0:,0]
        print(voltage.min(), voltage.max(), voltage.mean())
        return voltage

