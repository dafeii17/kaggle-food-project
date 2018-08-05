#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:17:17 2018

@author: yinglirao
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
import scipy

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

def create_dataset(error_sample_list, train_cuisine):
    new_x = np.empty((0, features.shape[1]))
    new_y = []
    for i in error_sample_list:
        new_x = scipy.sparse.vstack((new_x, x_train[train_cuisine==i]))
        new_y.extend(y_train[train_cuisine==i])
    return new_x, new_y

def create_new_testset(error_sample_list, pred_cuisine):
    x_new_test = np.empty((0, features.shape[1]))
    for i in error_sample_list:
        x_new_test = scipy.sparse.vstack((x_new_test, x_test[pred_cuisine==i]))
    return x_new_test

def update_ypreds(pred, error_sample_list,pred_cuisine,pred_origin):
    y_idx = 0
    for name in error_sample_list:
        check_idx = pred_cuisine==name
        for index, boolean in enumerate(check_idx):
            if boolean:
                pred_origin[index] = pred[y_idx]
                y_idx+=1
    return pred_origin

def model_correction(x_train, y_train, x_test, y_test):
    error_sample_lists = [['french', 'italian', 'southern_us'],
                      ['british', 'irish', 'french','southern_us'],
                      ['mexican','moroccan', 'spanish','greek'],
                      ['thai','indian', 'vietnamese'],
                      ['southern_us','italian'],
                      ['greek', 'british', 'spanish', 'irish','french','russian'],
                      ['japanese', 'chinese', 'filipino','korean','vietnamese']]
    models = [None]*len(error_sample_lists)
    preds = [None]*len(error_sample_lists)
    train_cuisine = le.inverse_transform(y_train)
    model_origin = LogisticRegression(random_state=0, penalty='l2', C=10)
    model_origin.fit(x_train, y_train)
    pred_origin = model_origin.predict(x_test)
    pred_cuisine = le.inverse_transform(pred_origin)
    
    for i in range(len(error_sample_lists)):
        models[i] = LogisticRegression(random_state=0, penalty='l2', C=10)
        new_x, new_y = create_dataset(error_sample_lists[i], train_cuisine)
        models[i].fit(new_x, new_y)
        test_new_x = create_new_testset(error_sample_lists[i], pred_cuisine)
        preds[i] = models[i].predict(test_new_x)
        pred_origin=update_ypreds(preds[i], error_sample_lists[i], pred_cuisine, pred_origin) 
    score = (pred_origin == y_test).sum()/len(y_test)         
    return pred_origin, score

data = load_data('../input/train.json')
ingredients, labels, le = extract_data(data)
features, tfidf = tfidf_transform(ingredients)
kf = KFold(n_splits=4)
score = np.zeros((1,4)).ravel()
for i, (train_idx, test_idx) in enumerate(kf.split(features)):
    print('train...')
    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    _, score[i] = model_correction(x_train, y_train, x_test, y_test)
#    model=LogisticRegression(random_state=0, penalty='l2', C=10)  #for logistic regression
#    model.fit(x_train, y_train)  #for logistic regression
#    score[i]=model.score(x_test, y_test) #for logistic regression
    print(score[i]) 
print('average cross validation sccore is: %.5f'%score.mean()) 
#0.78551 for model_correction
#0.78506 for base model LG C=10, l2 penalty
#0.78795 with logistic regression twice
    

    