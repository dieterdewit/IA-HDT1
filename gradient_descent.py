import numpy as np

def gradient_descent(
        X,
        y,
        theta_0,
        cost,
        cost_derivate,
        regParam,
        alpha,
        treshold,
        max_iter):
    theta, i = theta_0, 0
    costs = []
    gradient_norms = []
    while np.linalg.norm(cost_derivate(X, y, theta, regParam)) > treshold and i < max_iter:
        theta -= alpha * cost_derivate(X, y, theta, regParam)
        i += 1
        costs.append(cost(X, y, theta, regParam))
        gradient_norms.append(cost_derivate(X, y, theta, regParam))
    return theta, costs, gradient_norms

