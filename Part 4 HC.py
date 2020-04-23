#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:07:57 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values



# use the dendrogram 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))



from sklearn.cluster import AgglomerativeClustering

classifier = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')

Y = classifier.fit_predict(X)

plt.scatter(X[Y==0,0],X[Y==0,1], s=30, color = "red")
plt.scatter(X[Y==1,0],X[Y==1,1], s=30, color = "blue")
plt.scatter(X[Y==2,0],X[Y==2,1], s=30, color = "yellow")
plt.scatter(X[Y==3,0],X[Y==3,1], s=30, color = "green")
plt.scatter(X[Y==4,0],X[Y==4,1], s=30, color = "purple")
