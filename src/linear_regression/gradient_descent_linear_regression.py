from src.cost.cost_and_deriv import cost, cost_derivative
from src.utils.data_utils import load_data
from src.utils.visualize_utils import visualize_weights, line_plot, feature_vs_target

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def gradient_descent(fderiv, x, t, initial_start, step_size = 0.1, precision = 0.00001, max_iter = 1000):
    cur_stat = np.array(initial_start)
    last_stat = np.array([float('inf')] * len(initial_start))
    cost_list = []
    iter_list = []

    stat_list = [cur_stat]

    iter = 0
    
    while norm(cur_stat-last_stat) > precision and iter < max_iter:
        costi = cost(x, t, cur_stat)
        last_stat = cur_stat.copy()

        gs = fderiv(x, t, last_stat)
        gradient = np.array(gs)
        cur_stat -= gradient * step_size

        stat_list.append(cur_stat)
        iter += 1
        cost_list.append(costi)
        iter_list.append(iter)
    
    # Visualize iterations VS cost
    line_plot(iters=iter_list, cost=cost_list, title=f"Iterations VS Cost (lr={step_size} - precision={precision})", xlabel="Iterations", ylabel="Cost")

    min_cost = cost(x, t, cur_stat)
    print(f"Program Stops at Iteration #{iter}")
    print(f"Minimum Cost using this configuration (lr={step_size} and precision={precision}) = {min_cost}")
    return cur_stat, iter, min_cost


def simple_trial():
    # Input is 1D feature
    X = np.array([0, 0.2, 0.4, 0.8, 1])
    x = X.copy() # to use in visualization
    t = 5 + X # output linear, no noise

    X = X.reshape((-1, 1)) # let's reshape in 2D
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    weights = np.random.rand(X.shape[1]) # Initial params

    best_stat, iters, min_cost = gradient_descent(cost_derivative, X, t, weights)
    print(f"Best Parameters are:{best_stat}")
    visualize_weights(x, t, best_stat)


def train_linear():
    # Load the data
    x, t= load_data()

    x = np.hstack([np.ones((x.shape[0], 1)), x])

    # Initialize weights
    weights = np.array([1.0, 1.0, 1.0, 1.0]) # we have three features, so we will have 4 weights

    best_stat, iters, min_cost = gradient_descent(cost_derivative, x, t, weights, step_size=0.1, precision=0.00001, max_iter=3)
    print(f"Best Parameters are:{best_stat}")

def optimizing_hyperparams():
    lr_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001]
    precision_list = [0.01, 0.001, 0.0001, 0.00001]
    x, t = load_data()
    x = np.hstack([np.ones((x.shape[0], 1)), x])

    weights = np.array([1.0, 1.0, 1.0, 1.0])

    history = {}

    for lr in lr_list:
        for pr in precision_list:
            best_stat, iters, cost = gradient_descent(cost_derivative, x, t, weights, step_size=lr, precision=pr, max_iter=10000)
            history[cost] = [iters, lr, pr, best_stat]

    # Find Minimum Cost among all configurations
    min_cost = min(history)
    print(f"Minimum Cost among all Configuration: {min_cost}")
    print(f"Best Combination (lr & precsion) is: lr = {history[min_cost][1]} and precision = {history[min_cost][2]}")
    print(f"The Program with this Configuration Stops at number of Iteration = {history[min_cost][0]}")
    print(f"Best Weights: {history[min_cost][3]}")


def one_feature_regression(): 
    x, t, cols_names = load_data() 

    for i in range(x.shape[1]): 
        print(f"Working On Feature: {cols_names[i]}") 
        feature = x[:, i].reshape(-1, 1)   # make it 2D
        feature = np.hstack([np.ones((feature.shape[0], 1)), feature]) 
        feature_vs_target(feature[:, 1], t, 
                          title=f"{cols_names[i]} VS price", 
                          xlabel=f"x: {cols_names[i]}", 
                          ylabel="y: price") 

        weights = np.array([1.0, 1.0]) 
        gradient_descent(cost_derivative, feature, t, weights, 
                         step_size=0.001, precision=0.00001, max_iter=10000)



if __name__ == "__main__":
    # simple_trial()
    # train_linear()
    # optimizing_hyperparams()
    one_feature_regression()

# run file: python -m src.linear_regression.gradient_descent_linear_regression