#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:12:03 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labelencoder_1 = LabelEncoder()
labelencoder_2 = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [1])
X[:,1] = labelencoder_1.fit_transform(X[:,1])
X[:,2] = labelencoder_1.fit_transform(X[:,2])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#at this point, data is clean

import keras
from keras.models import Sequential
from keras.layers import Dense

#create the ANN, intiate! 
classifier = Sequential()
#add a input layer and the hidden lyaer (only 1 hidden layer)
#input is 11 dimensions, rule of thumb: first layer is 11=1/2 = 6
#init = how you initialize
#activation = relu, means rectify
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu',input_dim = 11))

classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'tanh',input_dim = 6))

#add the final layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid',input_dim = 4))

#compile the model, optimizer is called adam, loss is the loss function, metrics is the evaluation 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
"""
train it. also, we can add 2 more arguments: 1, batch size (# of observations
after which you adjust weights), epochL: number of times adjusting
"""
classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
A = confusion_matrix(y_pred,y_test)

accuracy = (A[0][0]+A[1][1])/2000