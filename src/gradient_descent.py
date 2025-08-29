import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Gradient Descent Function
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m = len(y)
    w, b = 0.0, 0.0
    cost_history = []

    for _ in range(epochs):
        y_pred = w * X + b
        dw = (-2/m) * np.sum(X * (y - y_pred))
        db = (-2/m) * np.sum(y - y_pred)
        w -= lr * dw
        b -= lr * db

        cost = np.mean((y - y_pred) ** 2)
        cost_history.append(cost)

    return w, b, cost_history

# Example Run
if __name__ == "__main__":
    X = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([3,5,7,9,11,13,15,17,19,21])

    w, b, cost_history = gradient_descent(X, y, lr=0.01, epochs=1000)
    print("Manual Gradient Descent → w:", w, ", b:", b)

    model = LinearRegression()
    model.fit(X.reshape(-1,1), y)
    print("Sklearn → w:", model.coef_[0], ", b:", model.intercept_)

    # Plot
    plt.plot(cost_history)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Cost Function Reduction")
    plt.savefig("../results/plots.png")
    plt.show()
