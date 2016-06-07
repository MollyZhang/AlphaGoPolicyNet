import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import go_parser


go_data = go_parser.parse_games(num_games="All", first_n_moves=1)

print "whatever"
