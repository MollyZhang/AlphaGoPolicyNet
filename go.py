# python libraries
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt

#my scripts
import go_parser as gp
import DNN_go_tensorflow as dnn_go
import visualization as vz


COLUMNS = [chr(ord('a') + i) for i in range(19)]
ROWS = [chr(ord('a') + i) for i in range(19)]

def main():
    plot_accuracy_scaling_with_training_example()


def plot_accuracy_scaling_with_training_example():
    accu = {"train": [], "test": []}
    for n in range(1000, 9000, 1000) + ['All']:
        go_data = gp.parse_games(num_games=n, first_n_moves=10, onehot=True)
        train_accuracy, test_accuracy = dnn_go.basic_3layer_NN(go_data, hidden_layer_num=2000)
        accu["train"].append(train_accuracy)
        accu["test"].append(test_accuracy)
        print n
        print train_accuracy
        print test_accuracy
    print accu
    with open("generated_data/first_10/sample_size_accuracy.pkl", "w") as f:
        f.write(pickle.dumps(accu))

def plot_hidden_node_and_accuracy():
    # accu = {"train": [], "test": []}
    # go_data = gp.parse_games(num_games=1000, first_n_moves=10, onehot=True)
    # for hidden_nodes in range(100, 1100, 100) + range(1500, 6500, 500):
    #     train_accuracy, test_accuracy = dnn_go.basic_3layer_NN(go_data, hidden_layer_num=hidden_nodes)
    #     accu["train"].append(train_accuracy)
    #     accu["test"].append(test_accuracy)
    #     print hidden_nodes
    #     print train_accuracy
    #     print test_accuracy
    # print accu
    # with open("generated_data/first_10/hidden_nodes_accuracy.pkl", "w") as f:
    #     f.write(pickle.dumps(accu))

    with open("generated_data/first_10/hidden_nodes_accuracy.pkl", "r") as f:
        accuracies = pickle.loads(f.read())
        f.close()

    x = range(100, 1100, 100) + range(1500, 6500, 500)

    plt.plot(x, accuracies["train"], "r")
    plt.plot(x, accuracies["test"], "g")
    legend=["train accuracy", "test accuracy"]
    plt.xlabel("number hidden nodes")
    plt.ylabel("accuracy")
    plt.title("prediction accuracy with different number of hidden nodes")
    plt.legend(legend, loc="best")
    plt.show()


def get_prediction_example():
    prob, board, move = gp.basic_3layer_NN(
        num_games=1000, first_n=10, epoch=20, move_only=True)
    print board
    print move
    print prob

    with open("probability_10_step", "w") as f:
        f.write(pickle.dumps([prob, board, move]))
        f.close()




def draw_board_probabilities_10_step():
    with open("probability_10_step", "r") as f:
        prob, board, move = pickle.loads(f.read())
        f.close()
    board = (board * 2).astype(int)

    vz.draw_board(board, move, prob)



def plot_accuracy_over_moves():
    with open("first_10_accuracies", "r") as f:
        accu = pickle.loads(f.read())

    plt.plot(accu)
    plt.xlabel("number of moves")
    plt.ylabel("prediction accuracy")
    plt.show()


def draw_board_probabilities():
    with open("probabilitiy_of_open_game", "r") as f:
        probs = pickle.loads(f.read())
        f.close()

    open_moves =gp.map_1d_to_2d(get_open_move())
    vz.draw_openning_board([], probs['All'])

def get_open_move():
    go_data = gp.parse_games(num_games=100, first_n_moves=1)
    open_moves = go_data.train.labels
    for i in [60, 72, 288, 300]:
        print i, len(open_moves[open_moves == i])
    return open_moves

if __name__ == '__main__':
    main()