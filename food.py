# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:07:23 2018

@author: YRao156839
"""

import pandas as pd
import json
import numpy as np
import os

from wordcloud import WordCloud
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

os.chdir('/Users/yrao156839/Downloads/food project')
with open('train.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

df.cuisine.value_counts().plot.bar()
plt.show()

df['cuisine_id'] = df['cuisine'].factorize()[0]
cuisine_id_df = df[['cuisine', 'cuisine_id']].drop_duplicates().sort_values('cuisine_id')
cuisine_to_id = dict(cuisine_id_df.values)
id_to_cuisine = dict(cuisine_id_df[['cuisine_id', 'cuisine']].values)

df['ingredients2'] = df['ingredients'].apply(lambda x: ' '.join(x))

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.ingredients2).toarray()
labels = df.cuisine_id


N = 4 #select the most correlated four ingredients
cuisine_list=[] #list of all cuisine names
com_unigram_list=[] #list of most correlated ingredients of each cuisine
for cuisine, cuisine_id in sorted(cuisine_to_id.items()):
    features_chi2 = chi2(features, labels == cuisine_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    #bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    cuisine_list.append(cuisine)
    com_unigram_list.append(unigrams[-N:])


f,axarr = plt.subplots(5, 4, figsize=(20,15), dpi=300)
wordcloud = []
def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = int(360.0 * 21.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

for i in range(len(com_unigram_list)):
    wordcloud.append(WordCloud(background_color='lightgrey',
                      width=1200,
                      height=800,
                color_func=random_color_func
                     ).generate(cuisine_list[i] +' ' + ' '.join(com_unigram_list[i])))
                               
    axarr[int(i/4), int(i%4)].imshow(wordcloud[i])
    axarr[int(i/4), int(i%4)].axis('off')
    
# Fine-tune figure; make subplots farther from each other.
f.subplots_adjust(hspace=0.3)
f.savefig('title.png')
plt.show()


#cross validation to select best model from LogisticRegression, SVC, NB
#model to predict cuisine name based on ingredients,

models = [
    LinearSVC(penalty='l2', C=1),
    LinearSVC(penalty='l2', C=10),
    LinearSVC(penalty='l2', C=100),
    MultinomialNB(),
    LogisticRegression(random_state=0, penalty='l1', C=1),
    LogisticRegression(random_state=0, penalty='l1', C=10),
    LogisticRegression(random_state=0, penalty='l2', C=1),   
    LogisticRegression(random_state=0, penalty='l2', C=10), 
    LogisticRegression(random_state=0, penalty='l2', C=100) 
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
model_name_list=['SVC-L2_C1', 'SVC-L2_C10', 'SVC-L2_C100', 
                 'NB', 'LG-L1_C1','LG-L1_C10','LG-L2_C1','LG-L2_C10','LG-L2_C100']
entries = []
for index, model in enumerate(models):
    #model_name = model.__class__.__name__
    model_name=model_name_list[index]
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
        

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.savefig('CV with Logistic regression-test1.png')
plt.show()


#best model is SVC and LogisticRegression(random_state=0, penalty='l2', C=10)

#confusion matrix from best model SVC
from sklearn.model_selection import train_test_split
model = LinearSVC(penalty='l2', C=1)
#model = LogisticRegression(random_state=0, penalty='l2', C=10)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(20,20), dpi=300)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=cuisine_id_df.cuisine.values, yticklabels=cuisine_id_df.cuisine.values)

ax.set_ylabel('Actual', fontsize=18)
ax.set_xlabel('Predicted', fontsize=18)
ax.tick_params(labelsize=16)
fig.savefig('confusion_matrix2.png')
plt.show()

#SVC modeling metrics
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['cuisine'].unique()))


#​#trying XGBoost to get better prediction
#X_train, X_test, y_train, y_test = train_test_split(df['ingredients2'], df['cuisine_id'], random_state = 0)
#y_train = y_train.values.reshape(-1,1)
#y_test = y_test.values.reshape(-1,1)
#​
#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#X_train_tfidf = tfidf.fit_transform(X_train).toarray()
#X_test_tfidf = tfidf.transform(X_test).toarray()
#​
#from sklearn.model_selection import KFold
#ntrain = X_train.shape[0]
#ntest = X_test.shape[0]
#SEED = 0 # for reproducibility
#NFOLDS = 5 # set folds for out-of-fold prediction
#kf = KFold(n_splits= NFOLDS)
#​
#def get_oof(clf, x_train, y_train, x_test):
#    oof_train = np.zeros((ntrain,))
#    oof_test = np.zeros((ntest,))
#    oof_test_skf = np.empty((NFOLDS, ntest))
#    i=0
#    for train_index, val_index in kf.split(x_train):
#        x_tr = x_train[train_index]
#        y_tr = y_train[train_index].ravel()
#        x_val = x_train[val_index]
#​
#        clf.fit(x_tr, y_tr)
#​
#        oof_train[val_index] = clf.predict(x_val)
#        oof_test_skf[i, :] = clf.predict(x_test)
#        i+=1
#​
#    oof_test[:] = oof_test_skf.mean(axis=0)
#    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#
#nb = MultinomialNB()
#nb_oof_train, nb_oof_test = get_oof(nb, X_train_tfidf, y_train.ravel(), X_test_tfidf)
#
#svc = LinearSVC()
#svc_oof_train, svc_oof_test = get_oof(svc, X_train_tfidf, y_train.ravel(), X_test_tfidf)
#
#lg = LogisticRegression(random_state=0, penalty='l2', C=10)
#lg_oof_train, lg_oof_test = get_oof(lg, X_train_tfidf, y_train.ravel(), X_test_tfidf)
#​
#
#x_train = np.concatenate(( nb_oof_train, svc_oof_train, lg_oof_train), axis=1)
#x_test = np.concatenate(( nb_oof_test, svc_oof_test, lg_oof_test), axis=1)
#
#
#base_predictions_train = pd.DataFrame( {'naive bayes': nb_oof_train.ravel(),
#     'svc': svc_oof_train.ravel(),
#     'logisic': lg_oof_train.ravel(),
#    })
#base_predictions_train.head()
#
#
#import seaborn as sns
#corr = base_predictions_train.corr()
#f, ax = plt.subplots(figsize=(10, 8))
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)
#plt.show()
#
#import xgboost as xgb
#gbm = xgb.XGBClassifier(
#    #learning_rate = 0.02,
# n_estimators= 2000,
# max_depth= 4,
# min_child_weight= 2,
# #gamma=1,
# gamma=0.9,                        
# subsample=0.8,
# colsample_bytree=0.8,
# objective= 'binary:logistic',
# nthread= -1,
# scale_pos_weight=1).fit(x_train, y_train.ravel())
#predictions = gbm.predict(x_test)
