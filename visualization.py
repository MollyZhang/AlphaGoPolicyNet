import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def draw_openning_board(stones, probs):
    # create a 8" x 8" board
    fig = plt.figure(figsize=[8,8])
    fig.patch.set_facecolor((1,1,.8))
    ax = fig.add_subplot(111)
    # draw the grid
    for x in range(19): ax.plot([x, x], [0,18], 'k')
    for y in range(19): ax.plot([0, 18], [y,y], 'k')

    # scale the axis area to fill the whole figure
    ax.set_position([0,0,1,1])
    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()
    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(-1,19)
    ax.set_ylim(-1,19)

    for stone in stones:
        print stone
        s1, = ax.plot(stone[1], 18-stone[0], 'o',markersize=28,
                      markeredgecolor=(0,0,0), markerfacecolor='k', markeredgewidth=2)

    #scale probabilities so that they look good of each stone
    Scaler = MinMaxScaler()
    new_probs = Scaler.fit_transform(np.log(probs))
    new_probs = new_probs/np.amax(new_probs)

    for i in range(0, 19):
        for j in range(0, 19):
            stone = (i, j)
            transparency = new_probs[stone]
            s, = ax.plot(stone[1], 18-stone[0], 'o', markersize=28,
                color = (0.75, 0, 0.75, transparency), markeredgewidth=0)

    plt.show()


def draw_board(board, move, prob):
    # create a 8" x 8" board
    fig = plt.figure(figsize=[8,8])
    fig.patch.set_facecolor((1,1,.8))

    ax = fig.add_subplot(111)
    for x in range(19):
        ax.plot([x, x], [0,18], 'k')
    for y in range(19):
        ax.plot([0, 18], [y,y], 'k')
    ax.set_position([0,0,1,1])
    ax.set_axis_off()
    ax.set_xlim(-1,19)
    ax.set_ylim(-1,19)

    color_dict = {1: "w", 2: "k"}

    # #scale probabilities so that they look good of each stone
    Scaler = MinMaxScaler()
    new_probs = Scaler.fit_transform(np.log(prob))
    new_probs = new_probs/np.amax(new_probs)


    for i in range(0, 19):
        for j in range(0, 19):
            stone = (i, j)
            # draw probability
            transparency = new_probs[stone]
            s, = ax.plot(stone[1], 18-stone[0], 'o', markersize=28,
                color = (0.75, 0, 0.75, transparency), markeredgewidth=0)

            if board[stone]:
                stone_color = color_dict[board[stone]]
                s1, = ax.plot(stone[1], 18-stone[0], 'o', markersize=28,
                    markerfacecolor=stone_color, markeredgewidth=1)

            # if move[stone] == 1:
            #     s1, = ax.plot(stone[1], 18-stone[0], 'o', markersize=28,
            #         markerfacecolor="r", markeredgewidth=1)





    plt.show()


    import matplotlib.pyplot as plt


def draw_neural_net(layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    '''
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9


    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size_a):
            for o in xrange(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
    ax.set_axis_off()
    plt.show()


