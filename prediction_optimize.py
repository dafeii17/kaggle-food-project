# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:54:56 2018

@author: YRao156839
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

def load_data(data_path):
    print('Loading data...')
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def extract_data(data):
    ingredients = [line['ingredients'] for line in data]
    ingredients =[' '.join(i) for i in ingredients] #convert lists of words to sentences
    cuisine = [line['cuisine'] for line in data]
    le = LabelEncoder()
    labels = le.fit_transform(cuisine) #convert cuisine to categorical labels
    return np.asarray(ingredients), np.asarray(cuisine), labels, le

def tfidf_transform(ingredients): 
    tfidf = TfidfVectorizer(min_df=5, binary=True,
                            ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(ingredients)
    return features, tfidf

data = load_data('./train.json')
ingredients, cuisine, labels, le = extract_data(data)
features, tfidf = tfidf_transform(ingredients)

from sklearn.model_selection import train_test_split
x_train, x_test, cuisine_train, cuisine_test = train_test_split(
            features, cuisine, test_size=0.2, random_state=0)
y_train = le.transform(cuisine_train)
y_test = le.transform(cuisine_test)

import scipy
error_sample_list = ['british', 'irish', 'french', 'spanish']
error_x = np.empty((0, features.shape[1]))
error_y = []
for i in error_sample_list:
    error_x = scipy.sparse.vstack((error_x, x_train[cuisine_train==i]))
    error_y.extend(y_train[cuisine_train==i])

model = LogisticRegression(random_state=0, penalty='l2', C=10)
model.fit(x_train, y_train)
model2 = LogisticRegression(random_state=0, penalty='l2', C=10)
model2.fit(error_x, error_y)

preds1 = model.predict(x_test)
preds1_cuisine = le.inverse_transform(preds1)
test_error_x = np.empty((0, features.shape[1]))

for i in error_sample_list:
    test_error_x = scipy.sparse.vstack((test_error_x, x_test[preds1_cuisine==i]))
#    length_cuisine.append(x_test[preds1_cuisine==i].shape[0])

preds2 = model2.predict(test_error_x)

y_idx = 0
for name in error_sample_list:
    check_idx = preds1_cuisine==name
    for index, boolean in enumerate(check_idx):
        if boolean == True:
            preds1[index] = preds2[y_idx]

from sklearn import metrics
print(metrics.classification_report(y_test, preds1, target_names=le.classes_))

