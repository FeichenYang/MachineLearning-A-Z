#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:08:40 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[2,3,4]].values

#elbow method to find the optimum k
from sklearn.cluster import KMeans

"""
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
"""


kmeans = KMeans(n_clusters = 6, init = 'k-means++')
kmeans.fit(X)

Y = kmeans.fit_predict(X)

plt.scatter(X[Y==0,0],X[Y==0,2], s=30, color = "red")
plt.scatter(X[Y==1,0],X[Y==1,2], s=30, color = "blue")
plt.scatter(X[Y==2,0],X[Y==2,2], s=30, color = "yellow")
plt.scatter(X[Y==3,0],X[Y==3,2], s=30, color = "green")
plt.scatter(X[Y==4,0],X[Y==4,2], s=30, color = "purple")
plt.scatter(X[Y==4,0],X[Y==5,2], s=30, color = "purple")
