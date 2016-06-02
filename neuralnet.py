import pandas as pd
import go_parser
import gzip, cPickle
import datetime
import tensorflow as tf



def main():
    train_data, val_data, test_data = \
        go_parser.parse_games(1000, test_percent=0.2, val_percent=0.2)


def comparing_data_loading_time():
    """
    comparing loading 1000 games with either cPickle or directly parsing from sgf
    result:
    time using cPickle:  0:00:28.862567
    time parsing from sgf 0:00:09.379862
    parsing directly wins...
    """
    t1 = datetime.datetime.now()
    train_data, val_data, test_data = load_data()
    t2 = datetime.datetime.now()
    t_delta1 = t2-t1

    t3 = datetime.datetime.now()
    train_data, val_data, test_data = \
        go_parser.parse_games(1000, test_percent=0.2, val_percent=0.2)
    t4 = datetime.datetime.now()
    t_delta2 = t4-t3
    print "time using cPickle: ", t_delta1
    print "time parsing from sgf", t_delta2


def save_data_sample():
    num_games = 1000
    train_data, val_data, test_data = \
        go_parser.parse_games(num_games, test_percent=0.2, val_percent=0.2)
    f = gzip.open('Sample_data/1000_games.pkl.gz', "wb")
    cPickle.dump([train_data, val_data, test_data], f)
    f.close()

def load_data():
    f = gzip.open('Sample_data/1000_games.pkl.gz', "rb")
    return cPickle.load(f)



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