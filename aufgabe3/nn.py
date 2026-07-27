import numpy as np
from aufgabe3.neuron import Neuron
from aufgabe2.ode_solver_runge_kutta import runge_kutta_method_matrix

class NN:
    def __init__(self, weights : np.ndarray, number_of_layers, neurons_in_layer):
        self.weights = weights
        self.weights_matrix = self._set_weights_matrix()
        self.board = None
        self.current_for_white = 15
        self.current_for_black = -8
        self.number_of_layers = number_of_layers
        self.neurons_in_layer = neurons_in_layer
        self.time_step = 0.01
        self.initial_state = np.array([-65, 0.3177, 0.0529, 0.5961])
        self.tuple_of_constants = (1, 36, 120, 0.3, -77, 50, -54.387)
        self.time_array = np.arange(0, 50, 0.01)

    def set_time_step(self, dt):
        self.time_step = dt

    def set_initial_state(self, initial_state: np.ndarray):
        self.initial_state = initial_state

    def set_tuple_of_constants(self, tuple_of_constants: tuple):
        self.tuple_of_constants = tuple_of_constants

    def set_time_array(self, time_array: np.ndarray):
        self.time_array = time_array

    def set_weights(self, weights : np.ndarray):
        self.weights = weights
        self.weights_matrix = self._set_weights_matrix()

    def set_currents_for_pixels(self, current_for_white, current_for_black):
        self.current_for_white = current_for_white
        self.current_for_black = current_for_black

    def input_board(self, board_input: np.ndarray):
        self.board = board_input

    def _set_weights_matrix(self) -> np.ndarray:
        w_12 = self.weights[0]
        w_34 = self.weights[1]
        w_15 = self.weights[2]
        w_25 = self.weights[3]
        w_35 = self.weights[4]
        w_45 = self.weights[5]
        return np.array([
            [0, 0, 0, 0, 0],
            [w_12, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, w_34, 0, 0],
            [w_15, w_25, w_35, w_45, 0]
                  ])


    def set_eigen_current_vector(self) -> np.ndarray:
        eigen_current = np.zeros(self.number_of_layers * self.neurons_in_layer+ 1)
        k = 0
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i, j] == 1:
                    eigen_current[k] = self.current_for_white
                if self.board[i, j] == 0:
                    eigen_current[k] = self.current_for_black
                k = k + 1
        return eigen_current

    def get_weights(self):
        return self.weights

    def get_state_matrix(self) -> np.ndarray:
        initial_state_matrix = np.tile(self.initial_state[:, None], (1, self.number_of_layers * self.neurons_in_layer + 1))
        state = runge_kutta_method_matrix(x_0 = initial_state_matrix,
                                          time_array=self.time_array,
                                          dt = self.time_step,
                                          eigen_current =self.set_eigen_current_vector(),
                                          weights_matrix = self.weights_matrix,
                                          tuple_of_constants = self.tuple_of_constants
                                          )

        return state

    def get_deciding_neuron_voltage(self) -> np.ndarray:
        return self.get_state_matrix()[:, 0, self.number_of_layers * self.neurons_in_layer]


