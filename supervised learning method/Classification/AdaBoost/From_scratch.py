from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import math


df = pd.read_csv("WRKSP-LR-II-DATA.csv")
X = df[['x','y']].values
y = df['l'].values
def adaboost(X,y):
    classfier = []
    restotal= []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(len(X_train))
    sign = np.zeros(len(y_train))
    sign2 = np.zeros(len(y_test))
    W = [1 / len(X_train)] * len(X_train)
    MAX = 0
    while MAX < 50:
            cls = DecisionTreeClassifier(max_depth=1).fit(X_train,y_train,sample_weight=W)
            predict = cls.predict(X_train)
            predicttest = cls.predict(X_test)
            e = 0
            for i in range(len(predict)):
                  if predict[i] != y_train[i]:
                     e += W[i]
            print(e)
            a = 1/2 * math.log((1-e)/e)
            result = [x * a for x in predict]
            result2 = [x * a for x in predicttest]
            classfier.append(result)
            restotal.append(result2)
            for i in range(len(W)):
               W[i] = W[i]* math.exp(-a*y_train[i]*predict[i])
            W = list(map(lambda x: x/sum(W),W))
            MAX = MAX + 1
    arr = np.array(classfier)
    arrtest = np.array(restotal)
    for ar in arr:
         sign = sign + ar
    for ar in arrtest:
         sign2 = sign2 + ar
    result = np.array([1 if x >= 0 else -1 for x in sign])
    result2 = np.array([1 if x >= 0 else -1 for x in sign2])
    acc = accuracy_score(y_train,result)
    acc2 = accuracy_score(y_test,result2)
    print("acc_test", acc2)
    print("acc_train", acc)
    return sign



print(adaboost(X,y))