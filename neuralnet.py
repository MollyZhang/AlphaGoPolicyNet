import pandas as pd
import go_parser
import tensorflow as tf


def main():
    train_data, val_data, test_data = \
        go_parser.parse_games(10, test_percent=0.2, val_percent=0.2)

    test_for_duplicate_move(train_data)


def test_for_duplicate_move(data):
    """ test for data correctness by asserting that a new stone
    has to be placed at an empty position on board """
    x = data[0]
    y = go_parser.map_1d_to_2d(data[1])

    for i in range(len(x)):
        print pd.DataFrame(x[i])
        print y[i]
        print "-------------------------------"
        assert(x[i][tuple(y[i])] == 0)

if __name__ == '__main__':
    main()