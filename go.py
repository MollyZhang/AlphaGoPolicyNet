import sys
import numpy as np
import pandas as pd

STONE_DICT = {"empty": 0, "white": 1, "black": 2}



def main():
    Game_Parser("Database/0196-1699/0196-00-00.sgf")



def Game_Parser(gamefile):
    """take a game in SGF format and convert it to 2 lists:
    first list: 2d matrix of 19 x 19, the shape of board at any given time (features)
    second list: the position of next move at any given time (label)
    """
    board_positions = []
    next_moves = []
    f = open(gamefile, "r")
    start_positions = np.zeros((19,19))
    for line in f:
        if line[0:3] == "AB[" or line[0:3] == "AW[":
            add_starting_stones(start_positions, line)
        elif line[0] == ";":
            board_positions.append(start_positions)
            pass

    print pd.DataFrame(start_positions)

def add_starting_stones(start_positions, line):
    if line[0:3] == "AB[":
        stone_type = "black"
    elif line[0:3] == "AW[":
        stone_type = "white"
    positions = line[3:-3].split("][")
    for p in positions:
        column = ord(p[0]) - ord('a')
        row = ord(p[1]) - ord('a')
        start_positions[column][row] = STONE_DICT[stone_type]
    return start_positions

if __name__ == '__main__':
    main()