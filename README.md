# food-project
Dataset source: https://www.kaggle.com/c/whats-cooking-kernels-only
Goal: Use recipe ingredients to categorize the cuisine

Test-Train split: 0.25
Classification models and accuracies:
Logistic Regression (penalty = 'l2', C=10)  (0.78506)
LinearSVC(C=1) (0.7848)
Feedforward Neural Network in pytorch (0.764)

0.78551 for model_correction (retrain cuisine classes with high correlation)
0.78795 with logistic regression twice (first logistic regression probability as input for 2nd logistic regression)
