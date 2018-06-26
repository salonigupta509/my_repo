# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 23:09:09 2018

@author: user pc
"""

import pandas as pd
data = pd.read_csv("Dota2data.txt")

features = data.iloc[:,:-1].values
labels = data.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder 

labelencoder = LabelEncoder()

for i in range(10):
    features[:,i] = labelencoder.fit_transform(features[:,i])

'''
features[:,0] = labelencoder.fit_transform(features[:,0])
features[:,1] = labelencoder.fit_transform(features[:,1]) 
features[:,2] = labelencoder.fit_transform(features[:,2])
features[:,3] = labelencoder.fit_transform(features[:,3]) 
features[:,4] = labelencoder.fit_transform(features[:,4])
features[:,5] = labelencoder.fit_transform(features[:,5]) 
features[:,6] = labelencoder.fit_transform(features[:,6])
features[:,7] = labelencoder.fit_transform(features[:,7]) 
features[:,8] = labelencoder.fit_transform(features[:,8])
features[:,9] = labelencoder.fit_transform(features[:,9]) 
'''

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train,labels_test = train_test_split(features,labels,test_size = 0.2, random_state = 0)

import numpy as np

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state = 0)
classifier.fit(features_train,labels_train)

labels_pred1= classifier.predict(features_test)
labels_pred2 = classifier.predict(np.array([63,81, 58, 74, 50, 91,0,0,0,0]).reshape(1,-1))

labels_pred3 = classifier.predict(np.array([0,81, 0, 0, 0, 0,0,0,0,0]).reshape(1,-1))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_pred1)

score = classifier.score(features_test,labels_test)
print score


# Visualising the Training set results
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
features_set, labels_set = features_train, labels_train
X1, X2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('')
plt.ylabel('Y')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
features_set, labels_set = features_test, labels_test
X1, X2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


