from src.cost.cost_and_deriv import cost, cost_derivative
from src.utils.data_utils import load_data

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def gradient_descent(fderiv, x, t, initial_start, step_size = 0.1, precision = 0.00001, max_iter = 1000):
    cur_stat = np.array(initial_start)
    last_stat = np.array([float('inf')] * len(initial_start))

    stat_list = [cur_stat]

    iter = 0
    
    while norm(cur_stat-last_stat) > precision and iter < max_iter:
        print(f"Iteration #{iter+1}:")
        print(f"Weights: {cur_stat}")
        print(f"Cost: {cost(x, t, cur_stat)}")
        last_stat = cur_stat.copy()

        gs = fderiv(x, t, last_stat)
        gradient = np.array(gs)
        print(f"Gradient: {gradient}\n")
        cur_stat -= gradient * step_size

        stat_list.append(cur_stat)
        iter += 1

    return stat_list, cur_stat


def simple_trial():
    # Input is 1D feature
    X = np.array([0, 0.2, 0.4, 0.8, 1])
    x = X.copy() # to use in visualization
    t = 5 + X # output linear, no noise

    X = X.reshape((-1, 1)) # let's reshape in 2D
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    weights = np.random.rand(X.shape[1]) # Initial params

    stat_list, best_stat = gradient_descent(cost_derivative, X, t, weights)
    print(f"Best Parameters are:{best_stat}")
    visualize(x, t, best_stat)


def visualize(x, t, w):
    w0, w1 = w

    # predicted line: y = w0 + w1 * x
    y_pred = w0 + w1 * x

    # Plot data and fitted line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, t, color="blue", label="Data Points")
    plt.plot(x, y_pred, color='red', linewidth=2, label=f"Fit: y = {w0:.2f} + {w1:.2f}x")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_linear():
    # Load the data
    x, t= load_data()

    x = np.hstack([np.ones((x.shape[0], 1)), x])

    # Initialize weights
    weights = np.array([1.0, 1.0, 1.0, 1.0]) # we have three features, so we will have 4 weights

    stat_list, best_stat = gradient_descent(cost_derivative, x, t, weights, step_size=0.1, precision=0.00001, max_iter=3)
    print(f"Best Parameters are:{best_stat}")

if __name__ == "__main__":
    train_linear()

# run file: python -m src.linear_regression.gradient_descent_linear_regression