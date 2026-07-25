import numpy as np
from aufgabe3.neuron import Neuron
from scipy.signal import find_peaks

"""
The module implements a neuronal network in the Hodgkin-Huxley model (HHM). The topology of the network is described in: aufgabe3.notebook.neuronales_netz.ipynb
"""
class NeuronalNetwork:
    """
    Class which represent a neuronal network in HHM. Its instance takes a board as 2x2 array, where 1 is white pixel and 0 is black pixel.
    An instance has an attribute of class Neuron that gives the final voltage, depending on what the network decides, whether the board is properly arranged.
    """
    def __init__(self, weights : np.ndarray, number_of_layers, neurons_in_layer):
        self.weights = weights
        self.board = None
        self.current_for_white = 20
        self.current_for_black = -5
        self.calculating_neuron = Neuron()
        self.number_of_layers = number_of_layers
        self.neurons_in_layer = neurons_in_layer

    def set_weights(self, weights : np.ndarray):
        self.weights = weights

    def set_currents_for_pixels(self, current_for_white, current_for_black):
        self.current_for_white = current_for_white
        self.current_for_black = current_for_black

    def _get_eigen_current(self, row, column):
        if self.board[row, column] != 1 and self.board[row, column] != 0:
            raise ValueError("Values for board elements are incorrect: it must be either 0 or 1")
        return self.current_for_white if self.board[row, column] == 1 else self.current_for_black

    def board_input(self, board_input: np.ndarray):
        self.board = board_input

    #the function calculates the currents in each single neuron and takes these value to find the voltage in the deciding neuron.
    def get_final_voltage(self) -> np.ndarray:
        rows, cols = self.weights.shape
        voltages = np.empty((self.number_of_layers, self.neurons_in_layer), dtype = object)


        for i in range(rows): #going through layers
            prev_voltages = []
            for j in range(cols): #going through a single layer i and saving voltages inside each neuron
                outer_current = np.zeros_like(self.calculating_neuron.time_array) #storage for the outer current applied to a single neuron
                for k in range(len(prev_voltages)): #calculating the outer current
                    outer_current = outer_current + self.weights[i,k] * prev_voltages[k]
                total_current = outer_current + self._get_eigen_current(row = i, column = j) #total current with consideration of the input to the cell
                voltage_i_j = self.calculating_neuron.get_voltage(total_current) #calculating the voltage inside the cell
                voltages[i,j] = voltage_i_j #saving the voltage of the cell
                prev_voltages.append(voltage_i_j)

        return self._sum_voltages(voltages) #calculating the voltage inside the last deciding cell

    #the function sums the voltages inside the receptor cells and sums them with the corresponding weights for the deciding cell
    def _sum_voltages(self, voltages : np.ndarray) -> np.ndarray:
        current = np.zeros_like(self.calculating_neuron.time_array)
        rows, cols = self.weights.shape
        for i in range(rows):
            for j in range(cols):
                current = current + self.weights[i,j] * voltages[i,j]
        return self.calculating_neuron.get_voltage(current)

    #the function test if the given input is properly arranged. The weight are predefined in: aufgabe3.notebook.neuronales_netz.ipynb. See the block, where the network run test for predefined weights.
    def give_answer(self) -> bool:
        voltage = self.get_final_voltage()
        # getting the time points, where the voltage spikes for voltage V >= 0
        peak_indices, _ = find_peaks(voltage, prominence=0.5, height=0)

        #getting the actual time values where the peaks occur
        peak_times = self.calculating_neuron.time_array[peak_indices]
        peaks_number = len(peak_times)

        #region where the specific peak occurs
        region_start = 20.0
        region_end = 30.0

        # checking if any peak falls within this region and there are two peaks in the whole time interval
        peaks_in_region = peak_times[(peak_times >= region_start) & (peak_times <= region_end)]
        has_peak = len(peaks_in_region) > 0
        if has_peak and peaks_number == 2:
            return True

        #another test for the second case
        region_start = 10.0
        region_end = 20.0
        peaks_in_region = peak_times[(peak_times >= region_start) & (peak_times <= region_end)]
        has_peak = len(peaks_in_region) > 0
        if has_peak:
            return True

        return False #otherwise the board is not properly arranged

