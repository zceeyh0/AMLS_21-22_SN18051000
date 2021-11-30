# Student Number: 18051000
# This file aims at building a SVM Classifier for classification tasks.
# A SVM Classifier is built to train and test data.
# Grid Search hyper-parameter tuning is used to find the optimal parameters.
# Cross Validation method is used to train models in multiple groups.


import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# build a SVM Classifier with hyper-parameter tuning and Cross Validation
def svm_classifier(x_train, y_train, x_test, y_test, cv_num):
    # build a model of SVM
    svm = SVC()
    # create a dictionary for hyper-parameter tuning
    # it stores all possible values for C (penalty of error),
    # gamma coefficients, and kernel functions
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['linear', 'rbf', 'poly']}
    start_t = time.time()
    # use GridSearchCV for hyper-parameter tuning and Cross Validation
    # cv_num is the number of data groups implementing Cross Validation
    # GridSearchCV returns a model with the optimal parameters
    svm_gscv = GridSearchCV(svm, param_grid, cv=cv_num)
    # fit the best model to the training data
    svm_gscv.fit(x_train, y_train)
    train_t = time.time()
    print('Time consumed for SVM training:',
          round(train_t - start_t, 2), 's')
    # predict the labels of test images with the best model
    y_pred1 = svm_gscv.predict(x_train)
    y_pred2 = svm_gscv.predict(x_test)
    pred_t = time.time()
    print('Time consumed for SVM predictions:',
          round(pred_t - train_t, 2), 's')
    # calculate the real accuracy score of this prediction
    train_score = metrics.accuracy_score(y_train, y_pred1)
    validate_score = metrics.accuracy_score(y_test, y_pred2)
    # return the best SVM estimator, the highest accuracy score
    # from Cross Validation, and the real prediction result
    return svm_gscv.best_params_, train_score, \
           svm_gscv.best_score_, validate_score
