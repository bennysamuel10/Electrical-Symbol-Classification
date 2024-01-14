import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

path = "C:/eletrical symbol classification/"
files = os.listdir(path)[:5]
print(files)

classes={'Ammeter':0,'diode':1,'cap':2,'curr_src':3,'voltmeter':4}

import cv2

X = []
Y = []

for cl in classes:
    pth = path+cl
    for img_name in os.listdir(pth):
        img = cv2.imread(pth+"/"+img_name,0)
        X.append(img)
        Y.append(classes[cl])
print("dataset created successfully!")

pd.Series(Y).value_counts()

X[0].shape

print(type(X))
X = np.array(X)
Y = np.array(Y)
print(type(X))

plt.imshow(X[200],cmap="gray")
print(Y[200])

X.shape

X_new = X.reshape(len(X),-1)
print(X_new.shape)
print(Y.shape)

120*120

print(X.shape)
print(X.ndim)
print(X_new.ndim)

xtrain, xtest, ytrain, ytest = train_test_split(X_new,Y,test_size=.20, random_state=10)

print(xtrain.shape, ytrain.shape)
print(xtest.shape,ytest.shape)

print(xtrain.max())
print(xtest.max())
x_train = xtrain/255
x_test = xtest/255
print(x_train.max())
print(x_test.max())

from sklearn.decomposition import PCA

print(x_train.shape, x_test.shape)
pca = PCA(.98)
xtrain = pca.fit_transform(x_train)
xtest = pca.transform(x_test)
print(xtrain.shape, xtest.shape)
print(pca.n_components)
print(pca.n_features_)

ytest[:10]

log = LogisticRegression()
log.fit(xtrain, ytrain)

tr_pred = log.predict(xtrain)
ts_pred = log.predict(xtest)

print("Training Score", accuracy_score(ytrain,tr_pred))
print("Testing Score",accuracy_score(ytest,ts_pred))

plt.imshow(x_test[0].reshape(120,120), cmap='gray')
print(ytest[0])

decode = {0:'Ammeter',1:'diode',2:'cap',3:'curr_src',4:'voltmeter'}
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_test[i].reshape(120,120),cmap='gray')
    plt.title(decode[ts_pred[i]])
    plt.axis('off')

np.where(ts_pred!=ytest)
d = pd.DataFrame({'Actual':ytest,'Prediction':ts_pred})
d[d['Actual']!=d['Prediction']]

img = cv2.resize(cv2.imread('2.bmp',0), (120,120))
plt.imshow(img, cmap='gray')

img = pca.transform(img.reshape(1,-1)/255)

decode[log.predict(img)[0]]