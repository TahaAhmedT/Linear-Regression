import matplotlib.pyplot as plt

def visualize_weights(x, t, w):
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


def line_plot(iters: list, cost: list, title: str = "", xlabel: str = "", ylabel: str = ""):
    plt.figure(figsize=(8, 6))
    plt.plot(iters, cost, linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()


def feature_vs_target(feature, target, title: str="", xlabel: str="", ylabel: str=""):
    plt.figure(figsize=(8, 6))
    plt.scatter(feature, target)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.show()

