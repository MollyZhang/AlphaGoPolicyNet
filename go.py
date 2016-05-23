import sys
import numpy as np
import pandas as pd
import glob
import os
import pickle



STONE_DICT = {"empty": 0, "W": 1, "B": 2}
STONE_DICT2 = {"Empty": 0, "Me": 1, "Opponent": 2}

COLUMNS = [chr(ord('a') + i) for i in range(19)]
ROWS = [chr(ord('a') + i) for i in range(19)]


def main():
    boards, moves = Game_Parser("fakegame1.sgf")




def parse_all_games():
    all_files = get_all_game_files()
    all_features = []
    all_labels = []
    for i in range(len(all_files)):
        print i
        print all_files[i]
        features, labels = Game_Parser(all_files[i])
        all_features += features
        all_labels += labels
    return all_features, all_labels


def get_all_game_files():
    folders = glob.glob("Database/*")
    game_files = []
    for folder in folders:
        if folder != "Non19x19Boards":
            game_files += glob.glob(folder + "/" + "*.sgf")
    return game_files

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
            add_starting_stones(start_positions, line)
        elif line[0] == ";" and ";B[" in line and ";W[" in line:
            all_moves += line.strip().strip().split(";")

    board_positions.append(start_positions)
    all_moves = filter(None, all_moves)  # remove empty string in list

    for move in all_moves:
        print pd.DataFrame(board_positions[-1], columns=COLUMNS, index=ROWS)
        print move
        print "------------------------"


        if "[tt]" in move:
            continue
        next_moves.append(parse_move(move))
        new_board_position = add_stone_to_board(board_positions[-1], next_moves[-1])
        board_positions.append(new_board_position)


    # remove last board position because there is no new moves
    board_positions.pop(-1)

    # stop distingush between black and white stone by
    # universally call the stones "my stones" and "oponent's stones"
    features, labels = universalize_stones(board_positions, next_moves)
    assert(len(features) == len(labels))
    return features, labels

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
    if board[row][column] != 0:
        # raise Exception("this position already has a stone!")
        # TODO: solve taking stone way
        pass
    else:
        board[row][column] = move[0]
    return board


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

if __name__ == '__main__':
    main()