#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:04:55 2020

@author: feichang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y.reshape(-1,1))

from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X,Y)

Y_pred = regressor.predict(X)

plt.scatter(X,Y,color = "red")
plt.plot(X,Y_pred,color = "blue")

Y_pred2 = regressor.predict(sc_X.transform([[6.5]]))
Y_actual = sc_Y.inverse_transform(Y_pred2)
