import numpy as np

def linear_cost(X, y, theta, regParam):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    reg = theta ** 2
    return (sq.sum() + np.sum(regParam * reg))/ (2 * m) 

