#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:48:20 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

# split train and test set, 20% for test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/3,random_state = 42)

#fit the shit
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
#在此，已经train过了regressor,它可以干各种事

y_pred = regressor.predict(X_test)


plt.scatter(X_test,Y_test, color = 'sienna')
plt.plot(X_test,y_pred, color = 'purple')
plt.title('Salary vs Experience')
plt.show()