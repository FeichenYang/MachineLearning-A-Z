#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:26:00 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

import math
d = 10
N = 10000
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
import random

for n in range(0,N):
    ad = 0
    max_random = 0
    #每个i，random draw一个theta
    for i in range(0,d):
        #bernoulli random draw
        random_draw = random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1)
        if random_draw > max_random:
            max_random = random_draw
            ad = i
    numbers_of_rewards_1[ad] += dataset.values[n,ad]
    numbers_of_rewards_0[ad]+= (1-dataset.values[n,ad])
    ads_selected.append(ad)
    total_reward += dataset.values[n,ad]
plt.hist(ads_selected[9000:])
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
        