"""
Animated plot for SVM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

from sklearn.svm import SVC

# plot decision region function
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
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X[y == 1, 0],
                X[y == 1, 1],
                c='b', marker='x',
                label='1')
    plt.scatter(X[y == -1, 0],
                X[y == -1, 1],
                c='r',
                marker='s',
                label='-1')

# generate data for plot
np.random.seed(0)
X_xor =  np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                      X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# ATTENTION:
# fix data, change the first two positions so the data
# always contains at least 2 unique y values
y_xor[0] = 1
y_xor[1] = -1

# generate graph for plot
fig, ax = plt.subplots()

# setup marker generator and color map
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:2])

# create a svc classifier using rbf kernel
svm = SVC(kernel='rbf', random_state=0, gamma=1, C=1)

def update(i):
    # erase the previous figure
    fig.clear()

    # update the data
    plot_decision_regions(X_xor[:i], y_xor[:i], classifier= svm)
    
    # redraw the plot
    plt.draw()


# Init only required for blitting to give a clean slate.
def init():
    plot_decision_regions(X_xor[:2], y_xor[:2], classifier=svm)

# Animate the plot
ani = animation.FuncAnimation(fig, update, np.arange(3, len(X_xor)), init_func=init,
                              interval=50, repeat=False)

# Show the canvas
plt.show()


