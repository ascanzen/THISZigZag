from thiszigzag import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = np.cumprod(1 + np.random.randn(100) * 0.01)
pivots = peak_valley_pivots(X, 0.03, -0.03)


def plot_pivots(X, pivots):
    plt.xlim(0, len(X))
    plt.ylim(X.min() * 0.99, X.max() * 1.01)
    plt.plot(np.arange(len(X)), X, "k:", alpha=0.5)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], "k-")
    plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color="g")
    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color="r")


plot_pivots(X, pivots)
plt.show()
