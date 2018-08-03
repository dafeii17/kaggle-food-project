#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:27:49 2018
@author: yinglirao
"""

import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
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
    return np.asarray(ingredients), labels, le

def tokenize(ingredients): #tokenize word counts in full ingredients list
    countv = CountVectorizer(min_df=5, binary=True,
                            ngram_range=(1, 2), stop_words='english')
    countv.fit_transform(ingredients)
    return countv

def tokenize_cuisine(countv): #tokenize word counts in each cuisine
    cuisine_word_counts = np.zeros(shape=(len(countv.vocabulary_), len(le.classes_)))
    for i in range(len(le.classes_)):
        cuisine_word_counts[:, i] = countv.transform(ingredients[labels==i]).mean(axis=0)
    return cuisine_word_counts

def plot_cuisine_corr(cuisine_word_counts):
    df = pd.DataFrame(data=cuisine_word_counts, columns=le.classes_)
    fig, ax =plt.subplots(figsize=(10,10), dpi=300)
    corr = df.corr()
    sns.heatmap(corr, 
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True,
                cbar_kws={"shrink": .82},
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    fig.savefig('mean_correlation.png')

data = load_data('./train.json')
ingredients, labels, le = extract_data(data) #extract data from json file
countv = tokenize(ingredients) #count tokenize all ingredients
cuisine_word_counts = tokenize_cuisine(countv) #average count token for each cuisine
plot_cuisine_corr(cuisine_word_counts) #correlation map among each cuisine









