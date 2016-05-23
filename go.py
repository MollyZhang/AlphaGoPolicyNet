import sys


def main():
    Game_Parser("Database/0196-1699/0196-00-00.sgf")



def Game_Parser(gamefile):
    """take a game in SGF format and convert it to
    an 3d matrix of 19 x 19 x steps
    """
    f = open(gamefile, "r")
    for line in f:
        sys.stdout.write(line)




if __name__ == '__main__':
    main()