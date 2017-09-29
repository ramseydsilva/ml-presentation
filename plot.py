from itertools import cycle


import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mimage
import matplotlib.cbook as cbook
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC


# setup marker generator and color map
MARKER = 'o'
COLORS = ['r', 'b', 'g']
CMAP = ListedColormap([COLORS[1], COLORS[0]])


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # fit the svm
    classifier.fit(X, y)

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=CMAP)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS[0], marker=MARKER, label='1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c=COLORS[1], marker=MARKER, label='-1')


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


def animate2D(df, factor=100.0):
    # ATTENTION:
    # dividing the data by a factor of 100
    # large numbers take a long time to compute for svm engine
    tempX = df.iloc[:, :2].transform(lambda x: x / factor)
    X_xor = tempX.iloc[:, :].values
    y_xor = df.iloc[:, 2].values

    # generate graph for plot
    fig, ax = plt.subplots()

    # create a svc classifier using rbf kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=1)

    def update(i):
        # erase the previous figure
        fig.clear()
        # update the data
        plot_decision_regions(X_xor[:i], y_xor[:i], classifier=svm)
        # redraw the plot
        plt.draw()

    # Animate the plot
    anim = animation.FuncAnimation(fig, update, np.arange(2, len(X_xor)), interval=1000,
                                   repeat=False)
    return anim


def plot_image(ax, image, extent=None, size=100):
    image = cbook.get_sample_data(image, asfileobj=False)
    image = mimage.imread(image)
    if not extent:
        extent = [-size, size, -size, size]
    ax.imshow(image, extent=extent, aspect='auto')
    ax.set_xlim([-size, size])
    ax.set_ylim([-size, size])
    ax.axis('off')


def plot_text(ax, text, size=1, **kwargs):
    ax.text(0, 0, text, **kwargs)
    ax.set_xlim([-size, size])
    ax.set_ylim([-size, size])
    ax.axis('off')
