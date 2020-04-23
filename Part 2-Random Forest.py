#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:40:00 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 3)
regressor.fit(X,Y)

plt.scatter(X,Y,color = "red")
plt.plot(X_grid,regressor.predict(X_grid), color = "blue")
plt.show()

Y_pred = regressor.predict([[6.5]])
