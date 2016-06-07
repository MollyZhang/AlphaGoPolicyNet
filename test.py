import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle

import go_parser
import DNN_go_tensorflow

def main():
    test_flattening_and_reshaping()



def test_flattening_and_reshaping():
    gamefile = "Database/1985/1985-07-23a.sgf"
    dummy1, dummy2, features, labels = go_parser.Game_Parser(gamefile, first_n_moves=100)
    for feature in features:
        feature1 = np.array(go_parser.map_2d_to_1d([feature], "x")[0]).reshape((19, 19))
        assert(np.array_equal(feature, feature1))
    for label in labels:
        flattened_label = go_parser.map_2d_to_1d([label], "y")[0]
        one_hot_flattened_label = go_parser.one_hot_encoding([flattened_label])[0]
        one_hot_flattened_label_as_board = one_hot_flattened_label.reshape((19, 19))
        assert(one_hot_flattened_label_as_board[label] == 1)




if __name__ == '__main__':
    main()
