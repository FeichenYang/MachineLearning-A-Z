#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:36:58 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk

#quoting = 3 means ignoring all the double quotes in the text
dataset = pd.read_csv("Restaurant_Reviews.tsv", sep = '\t', quoting = 3)

#Cleaning the data
#removed the punctuations, stemming - turn all tense of words to same word
"""
sub(pattern, repl, string, count = 0, flags = 0)
Pattern: '[^a-z]' means don't remove letters
Repl: replace the other stuff with ' '
"""

corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #lower case
    review = review.lower()
    #get rid of those generic words, like this, the, a, an. Import this stopword list
    from nltk.corpus import stopwords
    #split the reviews into words
    review = review.split()
    #if a word is in the list, pop it using list generator 
    # use the stopwords as a set, it's much faster
    review = [word for word in review if not word in set(stopwords.words('english'))]
    #stemming
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #joint the separate words back to sentence
    review = ' '.join(review)
    corpus.append(review)

#bag of words model! 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

classifier.fit(X_train,y_train)

y_predict = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
A = confusion_matrix(y_predict, y_test)

