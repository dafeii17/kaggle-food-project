# food-project
### Background:
I enjoy exploring different parts of world, enjoy the diverse nature, culture. As quoted from Kaggle, "Some of our strongest geographic and cultural associations are tied to a region's local foods". This Kaggle playground competitions asks one to predict the category of a dish's cuisine given a list of its ingredients. link to original dataset: [https://www.kaggle.com/c/whats-cooking-kernels-only](https://www.kaggle.com/c/whats-cooking-kernels-only)
### Exploratory data analysis:
This dataset contains around 50,000 rows of ingredients. I am interested to see the most distinct ingredients for different cuisines and the correlations between different cuisines.
1.	Extracted ingredient text feature with tf-idf tokenizer and analyzed the correlation among various cuisines based on the mean feature values of each cuisine.
2.	Analyzed the most discriminative ingredients of each cuisine based on Chi2 statistics. 
### Machine Learning models:
To predict the cuisine type based on list of ingredients, the best accuracies are obtained with feed-forward neural network in PyTorch and logistic regressor, and SVM RBF kernel in sklearn.
After analyzing the correlations between different cuisines, the accuracies could be further improved by retraining cuisine classes with high correlations. 
