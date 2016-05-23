import sys
import numpy as np
import pandas as pd

STONE_DICT = {"empty": 0, "W": 1, "B": 2}
STONE_DICT2 = {"Empty": 0, "Me": 1, "Opponent": 2}


def main():
    (features, labels) = Game_Parser("Database/0196-1699/0196-00-00.sgf")



def Game_Parser(gamefile):
    """take a game in SGF format and convert it to 2 lists:
    first list: 2d matrix of 19 x 19, the shape of board at any given time (features)
    second list: the position of next move at any given time (label)
    """
    board_positions = []
    all_moves = []
    next_moves = []
    f = open(gamefile, "r")
    start_positions = np.zeros((19,19))
    for line in f:
        if line[0:3] == "AB[" or line[0:3] == "AW[":
            add_starting_stones(start_positions, line)
        elif line[0] == ";":
            all_moves += line[1:].strip().strip().split(";")

    board_positions.append(start_positions)
    for move in all_moves:
        next_moves.append(parse_move(move))
        new_board_position = add_stone_to_board(board_positions[-1], next_moves[-1])
        board_positions.append(new_board_position)

    # remove last board position because there is no new moves
    board_positions.pop(-1)

    # stop distingush between black and white stone by
    # universally call the stones "my stones" and "oponent's stones"
    features, labels = universalize_stones(board_positions, next_moves)
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
        else:
            raise Exception("a move type can't be any number other than 1 or 2")
    return features, labels

def add_stone_to_board(board, move):
    column = move[1][0]
    row = move[1][1]
    if board[column][row] != 0:
        raise Exception("this position already has a stone!")
    else:
        board[column][row] = move[0]
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
        start_positions[column][row] = STONE_DICT[line[1]]
    return start_positions

if __name__ == '__main__':
    main()