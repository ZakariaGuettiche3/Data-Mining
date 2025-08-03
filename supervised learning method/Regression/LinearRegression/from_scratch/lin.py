import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../with_Sklearn/house (2).csv')
plt.scatter(df['surface'], df['loyer'], color='blue', alpha=0.5)
plt.xlabel('Surface (m²)')
plt.ylabel('Rent (€)')
plt.title('Surface vs Rent')
plt.show()
X = np.array(df['surface']).reshape(-1, 1)
y = np.array(df['loyer']).reshape(-1, 1)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  

theta = np.zeros((2, 1))  
learning_rate = 0.0001
n_iterations = 10000
m = len(y)


for i in range(n_iterations):
    gradients = (2 / m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

theta_0, theta_1 = theta[0, 0], theta[1, 0]
print(f"Learned parameters (Gradient Descent): θ₀ = {theta_0:.2f}, θ₁ = {theta_1:.2f}")
x_vals = np.linspace(0, 250, 100).reshape(-1, 1)
x_vals_b = np.c_[np.ones((x_vals.shape[0], 1)), x_vals]
y_vals = x_vals_b.dot(theta)

plt.scatter(X, y, color='blue', alpha=0.5, label='Actual data')
plt.plot(x_vals, y_vals, color='green')
plt.xlabel('Surface (m²)')
plt.ylabel('Rent (€)')
plt.title('Linear Regression via Gradient Descent')
plt.legend()
plt.show()
def predict_rent(x):
    return theta_0 + theta_1 * x

print(f"Predicted rent for 35 m²: {predict_rent(35):.2f} €")
