#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 01:06:03 2018

@author: yinglirao
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import KFold

def load_data(data_path):
    print('Loading data...')
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def extract_data(data, flag=None):
    ingredients = [line['ingredients'] for line in data]
    ingredients =[' '.join(i) for i in ingredients] #convert lists of words to sentences
    if flag == 'test':
        return np.asarray(ingredients)
    else:
        cuisine = [line['cuisine'] for line in data]
        le = LabelEncoder()
        labels = le.fit_transform(cuisine) #convert cuisine to categorical labels
        return np.asarray(ingredients), labels, le

def tfidf_transform(ingredients): 
    tfidf = TfidfVectorizer(min_df=5, binary=True,
                            ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(ingredients)
    return features, tfidf

def model_LG_twice(x_train, y_train, x_test, y_test):
    model = LogisticRegression(random_state=0, penalty='l2', C=10) 
    splits=10
    kf = KFold(n_splits=splits)
    
    x_train_val = np.zeros((x_train.shape[0], len(le.classes_)))
    x_test_val = np.zeros((splits, x_test.shape[0], len(le.classes_)))
    
    for index, (train_index, val_index) in enumerate(kf.split(x_train)):
        print('training... index: %.d'%index)
        x_train1 = x_train[train_index]
        y_train1 = y_train[train_index]
        x_val = x_train[val_index]
        model.fit(x_train1, y_train1)
        x_train_val[val_index, :] = model.predict_proba(x_val)
        x_test_val[index, :, :] = model.predict_proba(x_test)
    x_test_val2 = x_test_val.mean(axis=0)
    
    model2 = LinearSVC(penalty='l2', C=50) #best so far
    model2.fit(x_train_val, y_train)
    preds_final = model2.predict(x_test_val2)
    score = (preds_final == y_test).sum()/len(y_test)
    return preds_final, score

data = load_data('../input/train.json')
ingredients, labels, le = extract_data(data)
features, tfidf = tfidf_transform(ingredients)
kf = KFold(n_splits=4)
score = np.zeros((1,4)).ravel()
for i, (train_idx, test_idx) in enumerate(kf.split(features)):
    print('train...')
    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    _, score[i] = model_LG_twice(x_train, y_train, x_test, y_test)
    print(score[i])
print('average cross validation sccore is: %.5f'%score.mean()) 
#0.78795 with logistic regression twice



