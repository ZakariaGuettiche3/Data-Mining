import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import seaborn as sns

data = pd.read_csv("../with_sklearn/fruit_data_with_colors.txt", delimiter='\t')
data = data[['fruit_label', 'height', 'width']]
data['apple'] = (data['fruit_label'] == 1).astype(int)
data['mandarin'] = (data['fruit_label'] == 2).astype(int)
data['orange'] = (data['fruit_label'] == 3).astype(int)
data['lemon'] = (data['fruit_label'] == 4).astype(int)
plt.figure(figsize=(8, 6))
for label, color in zip([1, 2,3,4], ['red', 'gray','blue','green']):
    subset = data[data['fruit_label'] == label]
    fruit_name = {1: 'apple', 2: 'mandarin', 3: 'orange', 4: 'lemon'}[label]
    plt.scatter(subset['height'], subset['width'], label=fruit_name, c=color)
plt.xlabel('Height')
plt.ylabel('Width')
plt.legend()
plt.title('Fruits: ')
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * (y.T @ np.log(h ) + (1 - y).T @ np.log(1 - h))
    return cost

def gradient(theta, X, y):
    m = len(y)
    return (1/m) * X.T @ (sigmoid(X @ theta) - y)

X = data[['height', 'width']].values
y = data['apple'].values
y1 = data['mandarin'].values
y2 = data['orange'].values
y3 = data['lemon'].values
X = np.hstack([np.ones((X.shape[0], 1)), X])
initial_theta = np.zeros(X.shape[1])
opt_theta = fmin_tnc(func=cost_function, x0=initial_theta, fprime=gradient, args=(X, y))[0]
initial_theta = np.zeros(X.shape[1])
opt_theta1 = fmin_tnc(func=cost_function, x0=initial_theta, fprime=gradient, args=(X, y1))[0]
initial_theta = np.zeros(X.shape[1])
opt_theta2 = fmin_tnc(func=cost_function, x0=initial_theta, fprime=gradient, args=(X, y2))[0]
initial_theta = np.zeros(X.shape[1])
opt_theta3 = fmin_tnc(func=cost_function, x0=initial_theta, fprime=gradient, args=(X, y3))[0]
print("Theta optimisé :", opt_theta)
print("Coût :", cost_function(opt_theta, X, y))

def plot_decision_boundary(X, y, theta,name,couleur):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0][:,1], X[y==0][:,2], c='gray', label='Other')
    plt.scatter(X[y==1][:,1], X[y==1][:,2], c=couleur, label=name)
    x_values = [np.min(X[:,1]), np.max(X[:,1])]
    y_values = -(theta[0] + np.dot(theta[1], x_values)) / theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel("Height")
    plt.ylabel("Width")
    plt.legend()
    plt.title("Decision Boundary for Apple Classification")
    plt.show()

plot_decision_boundary(X, y, opt_theta,'appel','red')
plot_decision_boundary(X, y1, opt_theta1,'mandarin','blue')
plot_decision_boundary(X, y2, opt_theta2,'orange','green')
plot_decision_boundary(X, y3, opt_theta3,'lemon','black')
def predict_multiclass(X, all_theta):
    probs = np.array([sigmoid(X @ theta) for theta in all_theta])
    predictions = np.argmax(probs, axis=0) + 1  
    return predictions
all_theta = [opt_theta, opt_theta1, opt_theta2, opt_theta3]
y_true = data['fruit_label'].values  
y_pred = predict_multiclass(X, all_theta)

acc = accuracy_score(y_true, y_pred)
print(f" Exactitude multiclasse (one-vs-all) : {acc * 100:.2f} %")
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=['Apple','Mandarin','Orange','Lemon'],
            yticklabels=['Apple','Mandarin','Orange','Lemon'])
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.title('Matrice de confusion')
plt.show()


plt.figure(figsize=(10, 8))


colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'black'}
names = {1: 'apple', 2: 'mandarin', 3: 'orange', 4: 'lemon'}

for label in [1, 2, 3, 4]:
    subset = data[data['fruit_label'] == label]
    plt.scatter(subset['height'], subset['width'], c=colors[label], label=names[label])

for theta, color, name in zip(all_theta, ['red', 'blue', 'green', 'black'], names.values()):
    x_vals = np.array([X[:, 1].min(), X[:, 1].max()])
    y_vals = -(theta[0] + theta[1]*x_vals) / theta[2]
    plt.plot(x_vals, y_vals, linestyle='--', color=color, label=f'{name} boundary')

plt.xlabel("Height")
plt.ylabel("Width")
plt.title("Multiclass Decision Boundaries")
plt.legend()
plt.grid(True)
plt.show()
