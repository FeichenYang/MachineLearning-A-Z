#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 22:13:20 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

transactions = []

#建一个list，每个项是一个transaction。
for i in range(0,7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

results = list(rules)

print(results[0:5])


    