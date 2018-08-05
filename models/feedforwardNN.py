#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:27:49 2018

@author: yinglirao
"""

import json
import os
print(os.getcwd())



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

class Recipe_Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.len = len(features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    def __len__(self):
        return self.len

def load_data(data_path):
    print('Loading data...')
    with open(data_path, 'r') as f:
        data = json.load(f)
        ingredients = [' '.join(line['ingredients']) for line in data]
        cuisine = [line['cuisine'] for line in data]
        print (len(cuisine))
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', 
                                encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        features = tfidf.fit_transform(ingredients).toarray()
        le = LabelEncoder()
        labels = le.fit_transform(cuisine)
    return features, labels   
#        
def train(epoch):
    net.train()
    for batch_index, (features, labels) in enumerate(train_loader):       
        features_tensor = Variable(torch.DoubleTensor(features).float())
        labels_tensor = Variable(torch.LongTensor(labels))
    # Forward + Backward + Optimize
        optimizer.zero_grad() # zero the gradient buffer
        outputs = net(features_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
            
        if batch_index%100 == 0:
            print ('\nEpoch [%d], Step [%d/%d], Loss: %.4f'
                   %(epoch, batch_index*len(features), len(train_loader.dataset), 
                     loss.data[0].item()))
            test()

def predict(x_test):
    net.eval()
    test_tensor = Variable(torch.DoubleTensor(x_test).float())
    _, results = torch.max(net(test_tensor).data, 1)
    return results

def test():
    net.eval()
    correct=0
    for batch_index, (features, labels) in enumerate(test_loader):
        features_tensor = Variable(torch.DoubleTensor(features).float())
        labels_tensor = Variable(torch.LongTensor(labels))
        outputs = net(features_tensor)
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.eq(predicted, labels_tensor.view_as(predicted)).sum().data.tolist()
    print('Test set: average accuracy %.4f'% (correct/len(test_loader.dataset)))


features, labels = load_data('../input/train.json')
x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=0)

train_dataset = Recipe_Dataset(x_train, y_train)
test_dataset = Recipe_Dataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size= 60, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= 60, shuffle=False)

net = Net(input_size=17805, hidden_size=300, num_classes=20)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
 
epoch = 3
for i in range(epoch):
    train(i)

results = predict(x_test).numpy()
#0.764 feed forward neural network
