import numpy as np

from neuronal_network import NeuronalNetwork
import set_of_boards
from nn import NN


def get_random_board() -> tuple[np.ndarray, bool]:
    rand_int = np.random.randint(0,2)

    if rand_int == 0:     #chess board case
        chess_boards = set_of_boards.get_set_of_chess_boards()
        r = np.random.randint(0, len(chess_boards))
        return chess_boards[r], True
    if rand_int == 1:     #non chess board case
        non_chess_boards = set_of_boards.get_set_of_non_chess_boards()
        r = np.random.randint(0, len(non_chess_boards))
        return non_chess_boards[r], False
    return None, None


def loss_function(voltage: np.ndarray, is_chess_board : bool) -> float:
    voltage_max = np.max(voltage)

    if is_chess_board:
        return 0.0 if voltage_max >= 0 else -voltage_max
    else:
        return 0.0 if voltage_max < 0 else voltage_max

def learning(number_of_iterations, perturbation_factor,  number_of_layers,
             neurons_in_layer, bounce_for_parameters : float):
    k = (neurons_in_layer + 1) * neurons_in_layer / 2
    random_array = bounce_for_parameters * np.random.uniform(low=-1.0, high=1.0, size= int(number_of_layers * k))
    size = len(random_array)
    neuronal_network = NN(weights=random_array, number_of_layers=number_of_layers,
                               neurons_in_layer=neurons_in_layer)

    for iteration in range(number_of_iterations):
        board, target = get_random_board()
        neuronal_network.input_board(board)

        loss = loss_function(neuronal_network.get_deciding_neuron_voltage(), target)

        old_weights = neuronal_network.get_weights().copy()
        perturbation = np.random.normal(0, perturbation_factor, size= size)
        shifted_weights = neuronal_network.weights + perturbation
        shifted_weights = np.clip(shifted_weights, -bounce_for_parameters, bounce_for_parameters)
        neuronal_network.set_weights(shifted_weights)

        new_loss = loss_function(neuronal_network.get_deciding_neuron_voltage(), target)

        if new_loss >= loss:
            neuronal_network.set_weights(old_weights)


    np.save("optimal_weights.npy", neuronal_network.get_weights())


if __name__ == "__main__":
    learning(number_of_iterations=1000,
             perturbation_factor=0.01,
             number_of_layers=2,
             neurons_in_layer= 2,
             bounce_for_parameters = 0.15
             )






