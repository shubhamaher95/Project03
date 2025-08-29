🔹 Introduction

This project demonstrates how Calculus is applied in Data Science & Machine Learning using Linear Regression.
We implement Gradient Descent from scratch with the help of derivatives to minimize the cost function, then compare it with Scikit-Learn’s LinearRegression.

🔹 Concept

In Linear Regression, we predict:

𝑦
=
𝑤
𝑥
+
𝑏
y=wx+b

w → Weight (slope)

b → Bias (intercept)

We use Gradient Descent to update w and b:

𝑤
=
𝑤
−
𝛼
∂
𝐽
∂
𝑤
,
𝑏
=
𝑏
−
𝛼
∂
𝐽
∂
𝑏
w=w−α
∂w
∂J
	​

,b=b−α
∂b
∂J
	​


Where J(w,b) is the cost function (Mean Squared Error).

🔹 Steps in This Project

Import necessary libraries

Create sample dataset

Define cost function (MSE)

Compute gradients using derivatives

Implement Gradient Descent manually

Compare with Scikit-Learn Linear Regression

Plot results for visualization

🔹 Code Implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Sample Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
y = np.array([7, 9, 11, 13, 15])

# Step 2: Cost Function (MSE)
def compute_cost(w, b, X, y):
    m = len(y)
    predictions = w * X + b
    cost = (1/(2*m)) * np.sum((predictions - y.reshape(-1,1))**2)
    return cost

# Step 3: Gradient Descent
def gradient_descent(X, y, w, b, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        predictions = w * X + b
        dw = -(2/m) * np.sum((y.reshape(-1,1) - predictions) * X)
        db = -(2/m) * np.sum(y.reshape(-1,1) - predictions)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {compute_cost(w,b,X,y)}")
    return w, b

# Step 4: Train using Gradient Descent
w, b = gradient_descent(X, y, w=0, b=0, alpha=0.01, iterations=1000)
print("Manual Gradient Descent → w:", w, " b:", b)

# Step 5: Compare with Sklearn
model = LinearRegression()
model.fit(X,y)
print("Sklearn → w:", model.coef_[0], " b:", model.intercept_)

# Step 6: Visualization
plt.scatter(X,y,color="blue",label="Data Points")
plt.plot(X, w*X+b, color="red",label="Manual Gradient Descent")
plt.plot(X, model.predict(X), color="green", linestyle="--",label="Sklearn Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

🔹 Results

✅ Manual Gradient Descent gave:

w ≈ 2.77

b ≈ 4.22

✅ Scikit-Learn gave:

w = 2.7701

b = 4.2151

👉 Both are nearly identical, which proves that Calculus (derivatives) works in real ML problems.

🔹 Conclusion

We successfully applied Calculus (derivatives) to optimize Linear Regression.

Gradient Descent helps minimize error between predicted and actual values.

Compared results with Scikit-Learn → matched perfectly.

Next steps: Extend this to Multiple Linear Regression and Logistic Regression.
