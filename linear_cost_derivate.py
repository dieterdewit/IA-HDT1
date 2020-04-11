import numpy as np

def linear_cost_derivate(X, y, theta, regParam):
    h = np.matmul(X, theta)
    m, _ = X.shape
    reg = (regParam / m) * theta.sum()
    return ((np.matmul((h - y).T, X).T) + reg) / m

import numpy as np


