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


NUM_GAMES = 85931
STONE_DICT = {"empty": 0, "W": 1, "B": 2}
STONE_DICT2 = {"Empty": 0, "Me": 1, "Opponent": 2}

COLUMNS = [chr(ord('a') + i) for i in range(19)]
ROWS = [chr(ord('a') + i) for i in range(19)]



def parse_games(num_games, test_percent=0.2, val_percent=0.2, onehot=False):
    files = get_game_files(num_games=num_games)
    all_features = []
    all_labels = []
    for i in range(len(files)):
        print "parsing game", i, files[i]
        features, labels = Game_Parser(files[i])
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
    return train_data, val_data, test_data


def one_hot_encoding(y):
    onehot_y = []
    for each_y in y:
        vector = [0] * 361
        vector[each_y] = 1
        onehot_y.append(vector)
    return np.array(onehot_y)

def get_game_files(num_games="All"):
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


def Game_Parser(gamefile):
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
        if "[tt]" in move:
            continue
        next_moves.append(parse_move(move))
        new_board_position = add_stone_to_board(np.array(board_positions[-1], copy=True),
                                                next_moves[-1])
        board_positions.append(new_board_position)

    # remove last board position because there is no new moves
    board_positions.pop(-1)
    # stop distingush between black and white stone by
    # universally call the stones "my stones" and "oponent's stones"
    features, labels = universalize_stones(board_positions, next_moves)
    labels = [tuple((move[1], move[0])) for move in labels] # swtich column and row for future convenience

    oneD_features = map_2d_to_1d(features, "x")
    oneD_labels = map_2d_to_1d(labels, "y")



    assert(len(features) == len(oneD_labels))
    return oneD_features, oneD_labels

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