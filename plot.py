from itertools import cycle

import numpy as np

from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

MARKER = 'o'
COLORS = ['r', 'b', 'g']


def plot3D(*dfs, columns=None, figsize=(5, 5), plot_titles=False):
    """Plot a 3d graph using a set of dataframes"""
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

    return fig

def plot2D(*dfs, columns=None, figsize=(5, 5), plot_titles=False):
    """Plot a 2d graph using a set of dataframes"""
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

    return fig


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    colors = ('blue', 'red', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker='o', label=cl)
