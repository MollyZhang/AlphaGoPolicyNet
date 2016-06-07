# python libraries
import numpy as np
import tensorflow as tf
import pickle

#my scripts
import go_parser as gp
import DNN_go_tensorflow as dnn_go
import visualization as vz


def main():
    go_data = gp.parse_games(num_games='All', first_n_moves=1)
    with open("probabilitiy_of_open_game", "r") as f:
        probs = pickle.loads(f.read())
        f.close()
    open_moves = go_data.train.labels

    for i in [60, 72, 288, 300]:
        print i, len(open_moves[open_moves==i])

    vz.draw_board(gp.map_1d_to_2d(open_moves[:5]), probs['All'])



if __name__ == '__main__':
    main()