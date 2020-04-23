#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:17:25 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)

Y_pred = regressor.predict(X)

plt.scatter(X,Y,color = "red")

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.plot(X_grid, regressor.predict(X_grid), color = 'black')

Y_pred2 = regressor.predict([[6.5]])