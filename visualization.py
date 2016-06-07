import matplotlib.pyplot as plt
import numpy as np


def draw_board(stones, probs):
    # create a 8" x 8" board
    fig = plt.figure(figsize=[8,8])
    fig.patch.set_facecolor((1,1,.8))

    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(19):
        ax.plot([x, x], [0,18], 'k')
    for y in range(19):
        ax.plot([0, 18], [y,y], 'k')

    # scale the axis area to fill the whole figure
    ax.set_position([0,0,1,1])

    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()

    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(-1,19)
    ax.set_ylim(-1,19)

    for stone in stones:
        s1, = ax.plot(stone[1], 18-stone[0], 'o',markersize=28,
                      markeredgecolor=(0,0,0), markerfacecolor='w', markeredgewidth=2)

    #plot probabilities of each stone
    max_color = np.amax(probs)
    for i in range(0, 19):
        for j in range(0, 19):
            stone = (i, j)
            transparency = probs[stone]/max_color
            s, = ax.plot(stone[1], 18-stone[0], 'o',markersize=28,
                color=(0.75, 0, 0.75, transparency), markeredgewidth=0)








    plt.show()