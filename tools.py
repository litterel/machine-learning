import numpy as np
import matplotlib.pyplot as plt


def scatter_2d(X, y, alpha=0.6):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', alpha=alpha)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', alpha=alpha)


def plot_boundry(algo, axis, poly_algo=None, alpha=0.5):
    XX, YY = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)))
    plot_X = np.c_[XX.ravel(), YY.ravel()]
    if poly_algo is not None:
        poly_algo.fit(plot_X)
        plot_X = poly_algo.transform(plot_X)
    y = algo.predict(plot_X)
    y = y.reshape(XX.shape)
    plt.contourf(XX, YY, y, cmap=plt.cm.winter, alpha=alpha)

