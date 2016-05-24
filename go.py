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
    parse_all_games()



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

    my_stone_type = move[0]
    opponent_stone_type = [x for x in [1,2] if x != move[0]][0]
    if board[row][column] != 0:
        raise Exception("this position already has a stone!")
    else:
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

if __name__ == '__main__':
    main()