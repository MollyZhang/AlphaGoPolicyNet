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
    vz.draw_neural_net([10, 15, 10])


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