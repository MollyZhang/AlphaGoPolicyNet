import sys
import numpy as np
import pandas as pd

STONE_DICT = {"empty": 0, "W": 1, "B": 2}



def main():
    Game_Parser("Database/0196-1699/0196-00-00.sgf")



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
            board_positions.append(start_positions)
            all_moves += line[1:].strip().split(";")

    for move in all_moves:
        print parse_move(move)



    #print pd.DataFrame(start_positions)



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