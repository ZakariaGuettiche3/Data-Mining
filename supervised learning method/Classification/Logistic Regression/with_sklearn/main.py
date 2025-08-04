import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("fruit_data_with_colors.txt", delimiter='\t')

X = data[['height', 'width']].values
y = data['fruit_label'].values  


model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=400)
model.fit(X, y)


y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f" Exactitude du modèle multiclasse : {accuracy:.2f} %")


def plot_multiclass_boundary(X, y, model, title="Frontières de décision (Multiclasse)"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.rainbow, edgecolors='k')
    plt.xlabel("Height")
    plt.ylabel("Width")
    plt.title(title)
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.show()

plot_multiclass_boundary(X, y, model)
