import numpy as np

def cost(x, t, w):
    cost_w = (np.dot(x, w) - t) ** 2
    cost_w = (1 / (2 * x.shape[0])) * cost_w
    return np.sum(cost_w)


def cost_derivative(x, t, w):
    derivative_w = np.dot((np.dot(x, w) - t), x)
    return (1 / x.shape[0]) * derivative_w


def main():
    # Input is 1D feature
    X = np.array([0, 0.2, 0.4, 0.8, 1])
    t = 5 + X # output linear, no noise

    X = X.reshape((-1, 1)) # let's reshape in 2D
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    print(X.shape) # 5 x 2: for line mx + c

    weights = np.array([1.0, 1.0]) # Initial params

    print(cost(X, t, weights)) # cost: 8

    print(cost_derivative(X, t, weights)) # derivative: [-4.   -1.92]


if __name__ == "__main__":
    main()