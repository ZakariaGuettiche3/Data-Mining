from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
data = pd.read_csv('Heart.csv')
X = data.drop(columns=['HeartDisease']).values
Y = data['HeartDisease'].values


x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
acc = []


print(x_train)
def calculat_distance(X,Y):
    return np.sqrt(sum([(X[i] - Y[i])**2 for i in range(len(X))]))

for i in range(1,10):
    y_pred = []
    for xx_test in x_test:
        result = [calculat_distance(xx_test,x_train[j]) for j in range(0,len(x_train))]
        sorted_indices = np.argsort(result)
        sorted_indices = sorted_indices[:i]
        classe = [y_train[i] for i in sorted_indices]
        y_pred.append(Counter(classe).most_common(1)[0][0])
    acc.append(accuracy_score(y_test,y_pred))


plt.figure(figsize=(10, 6))
plt.plot(range(1,10),acc,marker='o')
plt.title("Accuracy as a Function of k in KNN")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.xticks(range(1,10))
plt.grid(True)
plt.show()

best_k = acc.index(max(acc))
print("max accuracy_score: ", max(acc))
print('with k :', best_k)
