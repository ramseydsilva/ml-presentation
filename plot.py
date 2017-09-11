from itertools import cycle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


MARKER = 'o'
COLORS = ['r', 'b', 'g']


def plot3D(*dfs, columns=None, figsize=(5, 5), plot_titles=False):

    # create matplotlib 3d axes
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig, azim=-115, elev=15)

    for df, color, in zip(dfs, cycle(COLORS)):
        X, Y, Z = (df[col] for col in columns)
        # plot hyperplane
        ax.scatter(X, Y, Z, c=color, marker=MARKER)

    # set axis labels
    for axis, col in zip(['x', 'y', 'z'], columns):
        getattr(ax, f'set_{axis}label')(col)

    if plot_titles:
        for df in dfs:
            for i, j, k, text in zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.index):
                corr = 2
                ax.text(i + corr, j + corr, k + corr, text)

    plt.show()


def plot2D(*dfs, columns=None, figsize=(5, 5), plot_titles=False):
    fig, ax = plt.subplots(figsize=figsize)

    for df, color in zip(dfs, cycle(COLORS)):
        X, Y = (df[col] for col in columns)
        plt.scatter(X, Y, c=color, marker=MARKER)

    for axis, col in zip(['x', 'y'], columns):
        getattr(ax, f'set_{axis}label')(col)

    if plot_titles:
        for df in dfs:
            for i, j, text in zip(df.iloc[:, 0], df.iloc[:, 1], df.index):
                corr = 2
                ax.annotate(text,  xy=(i + corr, j + corr))

    plt.show()
