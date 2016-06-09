"""
Import to notice: Go game SGF format specifies a stone position by "column, row",
however, numpy specifies a 2d array position by "row, column"
therefore the returned label is converted to the format of numpy convention "row, column" for downstream convenience
TAKEAWAY: everything is in "row, column" format
"""

import sys
import numpy as np
import pandas as pd
import glob
import os
import pickle
import tensorflow as tf


NUM_GAMES = 85931
STONE_DICT = {"empty": 0, "W": 1, "B": 2}
STONE_DICT2 = {"Empty": 0, "Me": 1, "Opponent": 2}

COLUMNS = [chr(ord('a') + i) for i in range(19)]
ROWS = [chr(ord('a') + i) for i in range(19)]

def prepare_data_sets(train_data, val_data, test_data, dtype=tf.float32):
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(train_data, dtype=dtype)
    data_sets.validation = DataSet(val_data, dtype=dtype)
    data_sets.test = DataSet(test_data, dtype=dtype)
    return data_sets


class DataSet(object):
    def __init__(self, data, dtype=tf.float32):
        """`dtype` can be either `uint8` to leave the input as `[0, 1, 2]`,
        or `float32` to rescale into `[0, 1]`."""
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
        if dtype == tf.float32:
            # Convert from [0, 1, 2] -> [0.0, 0.5, 1.0].
            features = data[0].astype(np.float32)
            features = np.multiply(features, 1.0 / 2.0)

        self._num_examples = data[1].shape[0]
        self._features = features
        self._labels = data[1]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]


def parse_games(num_games=1000, first_n_moves=10,
                test_percent=0.2, val_percent=0.2, onehot=False,
                lib="tensorflow", move_only=False):

    files = get_game_files(num_games=num_games, lib=lib)
    all_features = []
    all_labels = []
    for i in range(len(files)):
        if i % 1000 == 0:
            print "parsing game %d to %d" %(i, i+999)
        features, labels, dummy1, dummy2 = Game_Parser(files[i],
                                                       first_n_moves,
                                                       move_only)
        all_features += features
        all_labels += labels
    randomized_game_index = np.random.permutation(len(all_features))
    num_test = int(test_percent * len(all_features))
    num_val = int(val_percent * len(all_features))
    x_test = np.array(all_features)[randomized_game_index[:num_test]]
    y_test = np.array(all_labels)[randomized_game_index[:num_test]]
    x_val = np.array(all_features)[randomized_game_index[num_test: num_test+num_val]]
    y_val = np.array(all_labels)[randomized_game_index[num_test: num_test+num_val]]
    x_train = np.array(all_features)[randomized_game_index[num_test+num_val:]]
    y_train = np.array(all_labels)[randomized_game_index[num_test+num_val:]]

    if onehot:
        y_test = one_hot_encoding(y_test)
        y_val = one_hot_encoding(y_val)
        y_train = one_hot_encoding(y_train)

    train_data = (x_train, y_train)
    val_data = (x_val, y_val)
    test_data = (x_test, y_test)
    if lib == "tensorflow":
        go_data = prepare_data_sets(train_data, val_data, test_data)
        return go_data
    else:
        return train_data, val_data, test_data


def one_hot_encoding(y):
    onehot_y = []
    for each_y in y:
        vector = [0] * 361
        vector[each_y] = 1
        onehot_y.append(vector)
    return np.array(onehot_y)

def get_game_files(num_games="All", lib="tensorflow"):
    if lib == "tensorflow":
        folders = glob.glob("Database/*")
    else:
        folders = glob.glob("../Database/*")
    game_files = []
    for folder in folders:
        if folder != "Non19x19Boards":
            game_files += glob.glob(folder + "/" + "*.sgf")
    if num_games == "All":
        return game_files
    else:
        np.random.seed(0)
        return np.array(game_files)[np.random.permutation(NUM_GAMES)[:num_games]]


def Game_Parser(gamefile,first_n_moves, move_only=False):
    """take a game in SGF format and convert it to 2 lists:
    first list: 2d matrix of 19 x 19, the shape of board at any given time (features)
    second list: the position of next move at any given time (label)
    """
    board_positions = []
    all_moves = []
    next_moves = []
    f = open(gamefile, "r")
    start_positions = np.zeros((19, 19))
    for line in f:
        if line[0:3] == "AB[" or line[0:3] == "AW[":
            start_positions = add_starting_stones(start_positions, line)
        elif line[0] == ";" and ";B[" in line and ";W[" in line:
            all_moves += line.strip().strip().split(";")

    board_positions.append(start_positions)
    all_moves = filter(None, all_moves)  # remove empty string in list

    for move in all_moves:
        steps_taken = count_steps_taken(board_positions[-1])
        if steps_taken >= first_n_moves:
            if move_only:
                if len(board_positions) >= 2:
                    board_positions = [board_positions[-2]]
                    next_moves = [next_moves[-1]]
                else:
                    board_positions = []
                    next_moves = []
            break
        if "[tt]" in move:
            continue
        next_moves.append(parse_move(move))
        new_board_position = add_stone_to_board(np.array(board_positions[-1], copy=True),
                                                next_moves[-1])
        board_positions.append(new_board_position)

    # remove last board position because there is no new moves
    if not move_only:
        board_positions.pop(-1)
    # stop distingush between black and white stone by
    # universally call the stones "my stones" and "oponent's stones"
    features, labels = universalize_stones(board_positions, next_moves)
    labels = [tuple((move[1], move[0])) for move in labels] # swtich column and row for future convenience

    oneD_features = map_2d_to_1d(features, "x")
    oneD_labels = map_2d_to_1d(labels, "y")

    assert(len(features) == len(oneD_labels))
    return oneD_features, oneD_labels, features, labels


def count_steps_taken(board):
    copy_of_board = np.array(board, copy=True)
    copy_of_board[copy_of_board == 2] = 1
    return int(np.sum(copy_of_board))

def map_2d_to_1d(datas, data_type):
    """labels are tuples, which is hard to use as part of neural net, therefore
    I convert the label tuples (19x19 possible values) to a 1d array of lenght 19*19=361
    counted in the row-first-column-scecond order """
    if data_type == "y":
        return [label[0]*19+label[1] for label in datas]
    elif data_type == "x":
        return [np.array(feature).flatten() for feature in datas]
    else:
        raise Exception("only 'x' or 'y' can be passed as 2nd parameter in this function")



def map_1d_to_2d(labels):
    """the reversion of 1d label to 2d tuple label"""
    return [(label/19, label%19) for label in labels]


def universalize_stones(positions, moves):
    features = []
    labels = []
    for i in range(len(moves)):
        if moves[i][0] == 1:
            labels.append(moves[i][1])
            features.append(positions[i])
        elif moves[i][0] == 2:
            labels.append(moves[i][1])
            reversed_position = np.array(positions[i], copy=True)
            reversed_position[reversed_position == 1] = 10
            reversed_position[reversed_position == 2] = 1
            reversed_position[reversed_position == 10] = 2
            features.append(reversed_position)
        else:
            raise Exception("a move type can't be any number other than 1 or 2")
    return features, labels


def add_stone_to_board(board, move):
    column = move[1][0]
    row = move[1][1]
    my_stone_type = move[0]
    opponent_stone_type = [x for x in [1,2] if x != move[0]][0]

    if board[row][column] != 0:
        raise Exception("this position already has a stone!")
    else:
        # remove stones if the new stone closes the last eye of neibhour opponent stones
        board[row][column] = my_stone_type
        neighbors = get_neighbors_on_board((row, column))
        for neighbor in neighbors:
            if board[neighbor] == opponent_stone_type:
                board = remove_stones(board, neighbor, opponent_stone_type)
    return board


def remove_stones(board, stone, stone_type):
    stones_to_be_removed = get_connecting_stone_groups(board, stone, stone_type)
    eye_count = count_eyes(stones_to_be_removed, board, stone_type)
    if eye_count == 0:
        for stone in stones_to_be_removed:
            board[stone] = 0
    else:
        pass
    return board

def count_eyes(stones_to_be_removed, board, stone_type):
    eyes = []
    for stone in stones_to_be_removed:
        for each_neighor in get_neighbors_on_board(stone):
            if board[each_neighor] == 0:
                eyes.append(each_neighor)
    return len(set(eyes))



def get_connecting_stone_groups(board, stone, stone_type):
    stone_group = [stone]
    while True:
        new_stone_added = 0
        for stone in stone_group:
            neighbors = get_neighbors_on_board(stone)
            for neighbor in neighbors:
                if board[neighbor] == stone_type and neighbor not in stone_group:
                    stone_group.append(neighbor)
                    new_stone_added += 1
        if new_stone_added == 0:
            break
    return stone_group


def get_neighbors_on_board(stone):
    row, column = stone
    neighbors = []
    if row + 1 <= 18:
        neighbors.append((row+1, column))
    if row - 1 >= 0:
        neighbors.append((row-1, column))
    if column + 1 <= 18:
        neighbors.append((row, column+1))
    if column - 1 >= 0:
        neighbors.append((row, column-1))
    return neighbors


def parse_move(move):
    column = ord(move[2]) - ord('a')
    row = ord(move[3]) - ord('a')
    return (STONE_DICT[move[0]], (column, row))


def add_starting_stones(start_positions, line):
    positions = line[3:].strip()[0:-1].split("][")
    for p in positions:
        column = ord(p[0]) - ord('a')
        row = ord(p[1]) - ord('a')
        start_positions[row][column] = STONE_DICT[line[1]]
    return start_positions

def count_games_with_starting_stones():
    """result: 5192 games have starting stones
    """
    files = get_game_files(num_games='All')
    num = 0
    for gamefile in files:
        f = open(gamefile, "r")
        for line in f:
            if line[0:3] == "AB[" or line[0:3] == "AW[":
                num += 1
                break
    return num