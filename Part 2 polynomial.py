#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:51:26 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,-1].values

#dataset 太少不分testset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
#linear regression model trained

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
# to generate polynomial shit of X
X_poly = poly_reg.fit_transform(X)
#define poly regressor
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

Y_pred = lin_reg2.predict(X_poly)

plt.scatter(X,Y,color = "red")
plt.plot(X,Y_pred,color="blue")
plt.show()

lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

