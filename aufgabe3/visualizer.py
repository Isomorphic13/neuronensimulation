from idlelib import history

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from aufgabe3 import set_of_boards
from aufgabe3.neural_network import NeuralNetwork


def visualize_b():


    # 1. Create a Directed Graph
    G = nx.DiGraph()

    # 2. Define edges (simplified: no direct R_1 -> E or R_3 -> E)
    edges = [
        ("R_1", "R_2"),
        ("R_2", "E"),
        ("R_3", "R_4"),
        ("R_4", "E")
    ]
    G.add_edges_from(edges)

    # 3. Set manual positions
    pos = {
        "R_1": (0, 1),
        "R_2": (1, 1),
        "R_3": (0, -1),
        "R_4": (1, -1),
        "E": (2, 0)
    }

    # 4. Draw the graph
    plt.figure(figsize=(7, 3.5))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1")

    # Add LaTeX-formatted node labels
    labels = {"R_1": "$R_1$", "R_2": "$R_2$", "R_3": "$R_3$", "R_4": "$R_4$", "E": "$E$"}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    plt.axis('off')
    plt.show()

    # 1. Create a Directed Graph
    G = nx.DiGraph()

    # 2. Define edges (including R_3 connected to E)
    edges = [
        ("R_1", "R_2"),
        ("R_1", "R_5"),
        ("R_2", "R_5"),
        ("R_3", "R_4"),
        ("R_3", "R_5"),  # R_3 goes to E
        ("R_4", "R_5")
    ]
    G.add_edges_from(edges)

    # 3. Set manual positions
    pos = {
        "R_1": (0, 1),
        "R_2": (1, 1),
        "R_3": (0, -1),
        "R_4": (1, -1),
        "R_5": (2, 0)
    }

    # 4. Draw the graph
    plt.figure(figsize=(7, 3.5))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1")

    # Add LaTeX-formatted node labels
    labels = {"R_1": "$R_1$", "R_2": "$R_2$", "R_3": "$R_3$", "R_4": "$R_4$", "R_5": "$R_5$"}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    # Define edge weights/labels (w_{i5} for connections to E, and w_{34} for R_3 to R_4)
    edge_labels = {
        ("R_1", "R_2"): "$w_{12}$",
        ("R_1", "R_5"): "$w_{15}$",
        ("R_2", "R_5"): "$w_{25}$",
        ("R_3", "R_4"): "$w_{34}$",
        ("R_3", "R_5"): "$w_{35}$",
        ("R_4", "R_5"): "$w_{45}$"
    }

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11, label_pos=0.5)

    plt.axis('off')
    plt.show()


def visualize_b2(weights):
    network = NeuralNetwork(weights, 2, 2)
    network.set_time_array(np.arange(0,50, 0.01))
    boards = set_of_boards.get_set_of_boards()

    tpoints = network.time_array

    fig, ax = plt.subplots(len(boards), 1, figsize=(10, 24), sharex=True)
    ax = ax.flatten()
    fig.suptitle("Spannungsverlauf über Zeit sowie die Aussage des Model für alle Bretten.")
    fig.subplots_adjust(top=0.9, hspace=0.3)


    i = 0

    cb1 = set_of_boards.get_set_of_chess_boards()[0]
    cb2 = set_of_boards.get_set_of_chess_boards()[1]

    for b in boards:
        network.input_board(b)
        voltage = network.get_deciding_neuron_voltage()
        voltage_max = round(voltage.max(),2)
        voltage_mean = round(voltage.mean(), 2)
        board_label = (f"{'Schach' if np.array_equal(b,cb1) or np.array_equal(b,cb2) else ''}brett:\n{'▣' if b[0,0]== 1 else '▢'} {'▣' if b[0,1]== 1 else '▢'} \n{'▣' if b[1,0]== 1 else '▢'} {'▣' if b[1,1]== 1 else '▢'} \n"
                       + r"$V_{max}$ = " + f"{voltage_max}, " +r"$V_{mean}$ = " + f"{voltage_mean}")
        ax[i].plot(tpoints, voltage, label = board_label)
        ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        i += 1

    ax[-1].set_xlabel("Zeit, (ms)")

    plt.show()

def visualize_c(weights_list):
    weights = np.array(weights_list)
    network = NeuralNetwork(weights, 2, 2)
    network.set_time_array(np.arange(0, 20, 0.01))
    boards = set_of_boards.get_set_of_boards()

    tpoints = network.time_array

    fig, ax = plt.subplots(len(boards), 1, figsize=(10, 24), sharex=True)
    ax = ax.flatten()
    fig.suptitle("Spannungsverlauf über Zeit sowie die Aussage des Model für alle Bretten.")
    fig.subplots_adjust(top=0.9, hspace=0.3)

    i = 0

    cb1 = set_of_boards.get_set_of_chess_boards()[0]
    cb2 = set_of_boards.get_set_of_chess_boards()[1]

    for b in boards:
        network.input_board(b)
        voltage = network.get_deciding_neuron_voltage()
        voltage_max = round(voltage.max(), 2)
        voltage_mean = round(voltage.mean(), 2)
        board_label = (
                    f"{'Schach' if np.array_equal(b, cb1) or np.array_equal(b, cb2) else ''}Brett:\n{'▣' if b[0, 0] == 1 else '▢'} {'▣' if b[0, 1] == 1 else '▢'} \n{'▣' if b[1, 0] == 1 else '▢'} {'▣' if b[1, 1] == 1 else '▢'} \n"
                    + r"$V_{max}$ = " + f"{voltage_max}, " + r"$V_{mean}$ = " + f"{voltage_mean}")
        ax[i].plot(tpoints, voltage, label=board_label)
        ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        i += 1

    ax[-1].set_xlabel("Zeit, (ms)")

    plt.show()


def visualize_d(optimal_weights):
    weights = optimal_weights
    network = NeuralNetwork(weights, 2, 2)
    network.set_time_array(np.arange(0, 20, 0.01))
    boards = set_of_boards.get_set_of_boards()

    tpoints = network.time_array

    fig, ax = plt.subplots(len(boards), 1, figsize=(10, 24), sharex=True)
    ax = ax.flatten()
    fig.suptitle("Spannungsverlauf und Modellvorhersagen für alle Bretten")
    fig.subplots_adjust(top=0.9, hspace=0.3)

    i = 0

    cb1 = set_of_boards.get_set_of_chess_boards()[0]
    cb2 = set_of_boards.get_set_of_chess_boards()[1]

    for b in boards:
        network.input_board(b)
        voltage = network.get_deciding_neuron_voltage()
        voltage_max = round(voltage.max(), 2)
        voltage_mean = round(voltage.mean(), 2)
        board_label = (
                    f"{'Schach ' if np.array_equal(b, cb1) or np.array_equal(b, cb2) else ''}Brett:\n"
                    f"{'▣' if b[0, 0] == 1 else '▢'} {'▣' if b[0, 1] == 1 else '▢'} \n"
                    f"{'▣' if b[1, 0] == 1 else '▢'} {'▣' if b[1, 1] == 1 else '▢'} \n"
                    + "Aussage: " + f"{'Ja' if voltage_max >= 0 else 'Nein'}")
        ax[i].plot(tpoints, voltage, label=board_label)
        ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        i += 1

    ax[-1].set_xlabel("Zeit, ms")

    plt.show()


def visualize_d_corrected(optimal_weights):
    weights = optimal_weights
    network = NeuralNetwork(weights, 2, 2)
    network.set_time_array(np.arange(0, 20, 0.01))
    boards = set_of_boards.get_set_of_boards_two_black_pixels()

    tpoints = network.time_array

    fig, ax = plt.subplots(len(boards), 1, figsize=(10, 24), sharex=True)
    ax = ax.flatten()
    fig.suptitle("Spannungsverlauf und Modellvorhersagen für alle Bretter")
    fig.subplots_adjust(top=0.9, hspace=0.3)

    i = 0

    cb1 = set_of_boards.get_set_of_chess_boards()[0]
    cb2 = set_of_boards.get_set_of_chess_boards()[1]

    for b in boards:
        network.input_board(b)
        voltage = network.get_deciding_neuron_voltage()
        voltage_max = round(voltage.max(), 2)
        voltage_mean = round(voltage.mean(), 2)
        board_label = (
                    f"{'Schach' if np.array_equal(b, cb1) or np.array_equal(b, cb2) else ''}Brett:\n"
                    f"{'▣' if b[0, 0] == 1 else '▢'} {'▣' if b[0, 1] == 1 else '▢'} \n"
                    f"{'▣' if b[1, 0] == 1 else '▢'} {'▣' if b[1, 1] == 1 else '▢'} \n"
                    + "Aussage: " + f"{'Ja' if voltage_max >= 0 else 'Nein'}")
        ax[i].plot(tpoints, voltage, label=board_label)
        ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        i += 1

    ax[-1].set_xlabel("Zeit, ms")

    plt.show()


def visualize_error_history(error_history):
    tpoints = np.arange(0, len(error_history), 1)
    plt.figure(figsize=(10, 5))
    plt.plot(tpoints, error_history, linewidth=2)
    plt.xlabel("Anzahl der Iterationen")
    plt.ylabel("Fehler pro Iteration")
    plt.title("Verlauf von Fehler pro Iteration")
    plt.grid(True)
    plt.show


def visualize_weights_history(weights_history):

    # matrix shape: (num_time_moments, 6)
    matrix = np.array(weights_history)

    time = np.arange(matrix.shape[0])

    weight_labels = []
    weight_labels.append("$w_{12}$")
    weight_labels.append("$w_{34}$")
    weight_labels.append("$w_{15}$")
    weight_labels.append("$w_{25}$")
    weight_labels.append("$w_{35}$")
    weight_labels.append("$w_{45}$")

    plt.figure(figsize=(10, 6))
    for col in range(matrix.shape[1]):
        plt.plot(time, matrix[:, col], label=weight_labels[col])

    plt.xlabel("Anzahl der Iterationen")
    plt.ylabel("Gewichtungsgrößen")
    plt.title("Verlauf der Gewichtungsgrößen $w_{ij}$ während des Lernverfahrens.")
    plt.legend()
    plt.grid(True)
    plt.show()