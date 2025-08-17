import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.optimize import fmin_tnc

df = pd.read_csv('fruit_data_with_colors.txt', delimiter='\t')
X = df[['mass', 'width']].values
y = (df['fruit_label'] == 1).astype(int).values

# Standardize the features to help optimization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Map features to polynomial terms up to degree 6
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(X)

# 2. Logistic regression helpers

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-8  # to avoid log(0)
    cost = (-1/m) * (y.T.dot(np.log(h + epsilon)) +
                     (1 - y).T.dot(np.log(1 - h + epsilon)))
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return cost + reg

def gradient_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (1/m) * X.T.dot(h - y)
    grad[1:] += (lambda_ / m) * theta[1:]  # regularize except theta_0
    return grad

def train_logistic_regression(X, y, lambda_):
    initial_theta = np.zeros(X.shape[1])
    result = fmin_tnc(func=cost_function_reg,
                      x0=initial_theta,
                      fprime=gradient_reg,
                      args=(X, y, lambda_))
    return result[0]


# 3. Plot decision boundary

def plot_decision_boundary(X_original, y, lambda_, theta_optimal):
   
    x1_min, x1_max = X_original[:, 0].min(), X_original[:, 0].max()
    x2_min, x2_max = X_original[:, 1].min(), X_original[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                           np.linspace(x2_min, x2_max, 200))

  
    X_grid = np.c_[xx1.ravel(), xx2.ravel()]
    X_grid_scaled = scaler.transform(X_grid)
    X_grid_poly = poly.transform(X_grid_scaled)

    y_pred = sigmoid(X_grid_poly.dot(theta_optimal)).reshape(xx1.shape)

    positive = (y == 1)
    negative = (y == 0)
    plt.scatter(X_original[positive, 0], X_original[positive, 1],
                marker='o', label='Class 1')
    plt.scatter(X_original[negative, 0], X_original[negative, 1],
                marker='x', label='Class 0')

    plt.contour(xx1, xx2, y_pred, levels=[0.5], linewidths=2, colors='b')

    plt.xlabel('Mass')
    plt.ylabel('Width')
    plt.title(f'Decision Boundary (lambda = {lambda_})')
    plt.legend()
    plt.show()

# 4. Train and visualize for different lambdas

lambda_values = [0, 0.1, 1, 10]

# Use original (unscaled) X for plotting
X_original = df[['mass', 'width']].values

for lambda_ in lambda_values:
    print(f"\nTraining with lambda = {lambda_}...")
    theta_optimal = train_logistic_regression(X_poly, y, lambda_)
    plot_decision_boundary(X_original, y, lambda_, theta_optimal)
