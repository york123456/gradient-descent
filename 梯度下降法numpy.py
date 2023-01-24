import numpy as np

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = X @ theta
        error = h - y
        gradient = X.T @ error / m
        theta = theta - learning_rate * gradient
    return theta

# example usage
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([5, 7, 9, 11])
theta = np.array([0, 0])
learning_rate = 0.01
num_iterations = 1000

theta = gradient_descent(X, y, theta, learning_rate, num_iterations)
print(theta)
