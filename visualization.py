import matplotlib.pyplot as plt



def draw_board():
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
    # draw Go stones at (10,10) and (13,16)
    s1, = ax.plot(10,10,'o',markersize=28, markeredgecolor=(0,0,0), markerfacecolor='w', markeredgewidth=2)
    s2, = ax.plot(13,16,'o',markersize=28, markeredgecolor=(.5,.5,.5), markerfacecolor='k', markeredgewidth=2)

    plt.show()