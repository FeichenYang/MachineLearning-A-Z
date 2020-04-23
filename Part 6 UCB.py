#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:40:55 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing Random Selection
import random
N = 10000
d = 10

"""
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
"""
import math

numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(0, N):
    #翻的牌子是这样的，每个新用户都更新
    max_upper_bound = 0
    ad = 0
    #遍历每个广告，看一下他们的ub，来选
    for i in range(0,d):
            #对于每个ad，算一遍upper bound
        if (numbers_of_selections[i] > 0):      
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else: 
            #优先选没选过的
            upper_bound = 1e400
        #选出Upper confidance bound 广告
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    #选完妃之后，选择加一，reward加加。
    numbers_of_selections[ad] += 1
    sums_of_rewards[ad] += dataset.values[n,ad]
    ads_selected.append(ad)
    total_reward += dataset.values[n,ad]


# Visualising the results
plt.hist(ads_selected[9500:])
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()