import pandas as pd
import go_parser


x_train, x_val, x_test, y_train, y_val, y_test = \
    go_parser.parse_games(10, test_percent=0.2, val_percent=0.2)

# test for data correctness
for i in range(len(x_train)):
    column, row = y_train[i]
    assert(x_train[i][row, column] == 0)