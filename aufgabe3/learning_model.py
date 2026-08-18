import numpy as np
import set_of_boards
from neural_network import NeuralNetwork

def _loss_function(voltage: np.ndarray, is_chess_board : bool) -> float: #calculates loss for the given voltage function and target
    voltage_max = np.max(voltage)

    if is_chess_board:
        return 0.0 if voltage_max >= 0 else -voltage_max
    else:
        return 0.0 if voltage_max < 0 else voltage_max

def _get_mean_loss(neuronal_network : NeuralNetwork) -> float: #calculates the mean loss for all boards with the given weights, which are saved in 'neuronal_network'
    #mean loss for all chess boards
    chess_boards = set_of_boards.get_set_of_chess_boards()
    length = len(chess_boards)
    losses = np.empty(length)
    for i in range(length):
        neuronal_network.input_board(chess_boards[i])
        voltage = neuronal_network.get_deciding_neuron_voltage()
        loss = _loss_function(voltage, True)
        losses[i] = loss
    mean_loss_for_chess_boards = np.mean(losses)

    #mean loss for all non chessboard
    non_chess_boards = set_of_boards.get_set_of_non_chess_boards_two_black_pixels()
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
   # the model counts how many chess boards were predicted incorrectly with the weights that are saved in 'neural_network' as a parameter
  errors = 0

  chess_boards = set_of_boards.get_set_of_chess_boards()
  for board in chess_boards:
    neuronal_network.input_board(board)
    voltage_max = np.max(neuronal_network.get_deciding_neuron_voltage())
    prediction = voltage_max >= 0  # True if chess board, False otherwise
    if not prediction:
      errors += 1

  #same for the non chess boards
  non_chess_boards = set_of_boards.get_set_of_non_chess_boards_two_black_pixels()
  for board in non_chess_boards:
    neuronal_network.input_board(board)
    voltage_max = np.max(neuronal_network.get_deciding_neuron_voltage())
    prediction = voltage_max >= 0  # True if chess board, False otherwise
    if prediction:
      errors += 1

  #total error count for both types of boards
  return errors


def learn(number_of_iterations, perturbation_factor, number_of_layers,
             neurons_in_layer, bounce_for_parameters : float): #random work algorithm

    k = (neurons_in_layer + 1) * neurons_in_layer / 2
    random_array = bounce_for_parameters * np.random.uniform(low=-1.0, high=1.0, size=int(number_of_layers * k)) #creating random weights
    size = len(random_array)
    neuronal_network = NeuralNetwork(weights=random_array, number_of_layers=number_of_layers,
                                     neurons_in_layer=neurons_in_layer) #creating a neural network that does calculations with the current weights
    weights_history = []
    errors_history = []
    for iteration in range(number_of_iterations): #repeating learning process

        old_weights = neuronal_network.get_weights().copy()

        mean_loss = _get_mean_loss(neuronal_network)

        error = _get_incorrect_predictions_count(neuronal_network)
        errors_history.append(error)

        perturbation = np.random.normal(0, perturbation_factor, size=size)
        shifted_weights = old_weights + perturbation #slightly and randomly changing the weights values
        shifted_weights = np.clip(shifted_weights, -bounce_for_parameters, bounce_for_parameters) #clipping the weight values for numerical stability
        neuronal_network.set_weights(shifted_weights)

        new_mean_loss = _get_mean_loss(neuronal_network)

        if new_mean_loss >= mean_loss: #comparing which weights gives higher total mean loss value
            neuronal_network.set_weights(old_weights)

        weights_history.append(neuronal_network.get_weights().copy())

    np.save("notebook/weights_history_corrected.npy", np.array(weights_history).copy())
    np.save("notebook/errors_history_corrected.npy", np.array(errors_history).copy())
    np.save("notebook/optimal_weights_corrected.npy", neuronal_network.get_weights().copy())


if __name__ == "__main__": #running the script starts the learning process
    learn(number_of_iterations=800, perturbation_factor=0.01, number_of_layers=2, neurons_in_layer=2,
          bounce_for_parameters=0.15)






