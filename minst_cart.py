from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original',data_home='./datasets')
import numpy as np
import pandas as pd

### 用shuffle将数据集打散，并得到训练集和测试集
X = mnist.data
y = mnist.target
indice = np.array([i for i in range(X.shape[0])])
np.random.shuffle(indice)
X_train = X[indice[:int(X.shape[0]*0.8)]]
y_train = y[indice[:int(X.shape[0]*0.8)]]
X_test = X[indice[int(X.shape[0]*0.8)+1:]]
y_test = y[indice[int(X.shape[0]*0.8)+1:]]

### 训练模型
from sklearn import tree
tree_clf = tree.DecisionTreeClassifier(criterion='gini')
tree_clf.fit(X_train,y_train)
y_p = tree_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_p)) 

### 网格搜索
max_depth_list = [4,8,12]
min_samples_split_list = [2,4,8,12]
for max_depth in max_depth_list:
    for min_samples_split in min_samples_split_list:
        tree_clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=max_depth,min_samples_split=  
                                               min_samples_split)
        tree_clf.fit(X_train,y_train)
        y_p = tree_clf.predict(X_test)
        print(accuracy_score(y_test,y_p),max_depth,min_samples_split) 
