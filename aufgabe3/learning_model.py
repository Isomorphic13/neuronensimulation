import numpy as np
import set_of_boards
from neural_network import NeuralNetwork

def _loss_function(voltage: np.ndarray, is_chess_board : bool) -> float:
    voltage_max = np.max(voltage)

    if is_chess_board:
        return 0.0 if voltage_max >= 0 else -voltage_max
    else:
        return 0.0 if voltage_max < 0 else voltage_max

def _get_mean_loss(neuronal_network : NeuralNetwork) -> float:
    chess_boards = set_of_boards.get_set_of_chess_boards()
    length = len(chess_boards)
    losses = np.empty(length)
    for i in range(length):
        neuronal_network.input_board(chess_boards[i])
        voltage = neuronal_network.get_deciding_neuron_voltage()
        loss = _loss_function(voltage, True)
        losses[i] = loss
    mean_loss_for_chess_boards = np.mean(losses)

    non_chess_boards = set_of_boards.get_set_of_non_chess_boards()
    length = len(non_chess_boards)
    losses = np.empty(length)
    for i in range(length):
        neuronal_network.input_board(non_chess_boards[i])
        voltage = neuronal_network.get_deciding_neuron_voltage()
        loss = _loss_function(voltage, False)
        losses[i] = loss
    mean_loss_for_non_chess_boards = np.mean(losses)

    return float(mean_loss_for_chess_boards + mean_loss_for_non_chess_boards)

def _get_incorrect_predictions_count(neuronal_network: NeuralNetwork) -> int:

  errors = 0

  chess_boards = set_of_boards.get_set_of_chess_boards()
  for board in chess_boards:
    neuronal_network.input_board(board)
    voltage_max = np.max(neuronal_network.get_deciding_neuron_voltage())
    prediction = voltage_max >= 0  # True if chess board, False otherwise
    if not prediction:
      errors += 1


  non_chess_boards = set_of_boards.get_set_of_non_chess_boards()
  for board in non_chess_boards:
    neuronal_network.input_board(board)
    voltage_max = np.max(neuronal_network.get_deciding_neuron_voltage())
    prediction = voltage_max >= 0  # True if chess board, False otherwise
    if prediction:
      errors += 1

  return errors


def learn(number_of_iterations, perturbation_factor, number_of_layers,
             neurons_in_layer, bounce_for_parameters : float):
    k = (neurons_in_layer + 1) * neurons_in_layer / 2
    random_array = bounce_for_parameters * np.random.uniform(low=-1.0, high=1.0, size=int(number_of_layers * k))
    size = len(random_array)
    neuronal_network = NeuralNetwork(weights=random_array, number_of_layers=number_of_layers,
                                     neurons_in_layer=neurons_in_layer)
    weights_history = []
    errors_history = []
    for iteration in range(number_of_iterations):

        old_weights = neuronal_network.get_weights().copy()

        mean_loss = _get_mean_loss(neuronal_network)

        error = _get_incorrect_predictions_count(neuronal_network)
        errors_history.append(error)

        perturbation = np.random.normal(0, perturbation_factor, size=size)
        shifted_weights = neuronal_network.weights + perturbation
        shifted_weights = np.clip(shifted_weights, -bounce_for_parameters, bounce_for_parameters)
        neuronal_network.set_weights(shifted_weights)

        new_mean_loss = _get_mean_loss(neuronal_network)

        if new_mean_loss >= mean_loss:
            neuronal_network.set_weights(old_weights)

        weights_history.append(neuronal_network.get_weights().copy())

    np.save("optimal_weights.npy", neuronal_network.get_weights().copy())
    np.save("notebook/errors_history.npy", np.array(errors_history))


if __name__ == "__main__":
    learn(number_of_iterations=100, perturbation_factor=0.01, number_of_layers=2, neurons_in_layer=2,
          bounce_for_parameters=0.15)






