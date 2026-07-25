import numpy as np

from aufgabe3.notebook.neuronales_netz import neuronal_network
from neuronal_network import NeuronalNetwork
import set_of_boards

class LearningModel:

    def __init__(self):
        random_array = np.random.rand(2, 2)
        self.neuronal_network = NeuronalNetwork(weights = random_array, number_of_layers= 2, neurons_in_layer = 2)


    def get_random_board(self) -> np.ndarray:
        rand_int = np.random.randint(0,2)

        if rand_int == 0:     #chess board case
            chess_boards = set_of_boards.get_set_of_boards()
            r = np.random.randint(0, len(chess_boards))
            return chess_boards[r]
        if rand_int == 1:     #non chess board case
            non_chess_boards = set_of_boards.get_set_of_boards()
            r = np.random.randint(0, len(non_chess_boards))
            return non_chess_boards[r]


    def predict(self, input : np.ndarray) -> bool:
        self.neuronal_network.board_input(input)
        time_array = self.neuronal_network.calculating_neuron.time_array
        voltage = self.neuronal_network.get_final_voltage()




