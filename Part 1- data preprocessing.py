# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#take care to missing data
from sklearn.preprocessing import Imputer
# create imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'median', axis = 0)
#fit imputer to matrix
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#dummy variable for country name.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

#encoding catoagorical variables for yes or no
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# split train and test set, 20% for test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
