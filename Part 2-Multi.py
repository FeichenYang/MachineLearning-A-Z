#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:58:03 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#先labelencoder,然后onehotencoder. 用categorical features定义code的位置
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoid the dummy variable trap (the first 3 colummns are linearly dependant)
#remove the first column
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#同一个library，apply to 向量就是单变量，to matrix就是多变量
regressor.fit(X_train,Y_train)
#now trained.

#Predict
y_pred = regressor.predict(X_test)

import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#把每个index of columns into X
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#train了这个regressor

regressor_OLS.summary()
#发现2的pvalue 最大

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#train了这个regressor
regressor_OLS.summary()
#发现x2的p value最大

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#train了这个regressor
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#train了这个regressor
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#train了这个regressor
regressor_OLS.summary()

#结论：R&D投入越大，profit越大。其他基本没用。

