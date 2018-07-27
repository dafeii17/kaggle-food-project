#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:27:49 2018

@author: yinglirao
"""

import pandas as pd
import json
import numpy as np
import os
import sys

print(os.getcwd())

from wordcloud import WordCloud
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
 def __init__(self, input_size, hidden_size, num_classes):
     super(Net, self).__init__()
     self.layer_1 = nn.Linear(input_size,hidden_size, bias=True)
     self.relu = nn.ReLU()
     self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
     self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

 def forward(self, x):
     out = self.layer_1(x)
     out = self.relu(out)
     out = self.layer_2(out)
     out = self.relu(out)
     out = self.output_layer(out)
     return out
 


def load_data(data_path):
    print('Loading data...')
    with open(data_path, 'r') as f:
        data = json.load(f)
        ingredients = [' '.join(line['ingredients']) for line in data]
        cuisine = [line['cuisine'] for line in data]
        print (len(cuisine))
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        features = tfidf.fit_transform(ingredients).toarray()
        feature_names = tfidf.get_feature_names()
        
        le = LabelEncoder()
        labels = le.fit_transform(cuisine)

    return features, labels
 
features, labels = load_data('/Users/yinglirao/Desktop/food projects/train.json')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.33, random_state=0)

nets = [Net(input_size=17805, hidden_size=500, num_classes=20),
           Net(input_size=17805, hidden_size=1000, num_classes=20)]
criterion = nn.CrossEntropyLoss()
optimizers = [torch.optim.Adam(nets[0].parameters()), 
             torch.optim.Adam(nets[1].parameters())]

def get_batch(features, labels, i, batch_size):
    batch_x = features[i*batch_size: i*batch_size + batch_size]
    batch_y = labels[i*batch_size: i*batch_size + batch_size]
    return batch_x, batch_y


def optimization(net, optimizer):
    num_epochs = 20
    batch_size = 50
    total_batch = int(len(x_train)/batch_size)
    
    for epoch in range(num_epochs):
        for i in range(total_batch):
            batch_x,batch_y = get_batch(x_train, y_train, i,batch_size)
            features_tensor = Variable(torch.FloatTensor(batch_x))
            labels_tensor = Variable(torch.LongTensor(batch_y))
    
            # Forward + Backward + Optimize
            optimizer.zero_grad() # zero the gradient buffer
            outputs = net(features_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
            
            if (i+1)%40 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
     				%(epoch+1, num_epochs, i+1, total_batch, loss.data[0]))
    

    return net


test_feature_tensor = Variable(torch.FloatTensor(x_test))
test_tensor = Variable(torch.LongTensor(y_test))

        
 
results_sum=[]
for i in range(len(nets)):
    nets[i]= optimization(nets[i], optimizers[i])
    predict_outputs = nets[i](test_feature_tensor)
    _, predicted = torch.max(predict_outputs.data, 1)
    total = test_tensor.size(0)
    correct = (predicted == test_tensor).sum().item()
    print('Accuracy of the network on the test_set is %.4f%%' % (100 * correct/total))    
    results_sum.append(correct/total)
    
    
    