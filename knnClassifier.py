# Student Number: 18051000
# This file builds a KNN Classifier for classification tasks.
# A KNN Classifier is built to train and test data.
# Grid Search hyper-parameter tuning is used to find the optimal parameters.
# Cross Validation method is used to train models in multiple groups.


import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# build a KNN Classifier with hyper-parameter tuning and Cross Validation
def knn_classification(x_train, y_train, x_test, y_test, cv_num=5):
    # build a model of KNN with default parameters
    knn = KNeighborsClassifier()
    # create a dictionary of all values for n_neighbors
    param_grid = {'n_neighbors': np.arange(3, 50)}
    start_t = time.time()
    # use GridSearchCV for hyper-parameter tuning and Cross Validation
    # cv_num is the number of data groups implementing Cross Validation
    # GridSearchCV returns a model with the optimal parameters
    knn_gscv = GridSearchCV(knn, param_grid, cv=cv_num)
    # fit the best model to the training data
    knn_gscv.fit(x_train, y_train)
    train_t = time.time()
    print('Time consumed for KNN training:',
          round(train_t - start_t, 2), 's')
    # predict the labels of test images with the best model
    y_pred1 = knn_gscv.predict(x_train)
    y_pred2 = knn_gscv.predict(x_test)
    pred_t = time.time()
    print('Time consumed for KNN predictions:',
          round(pred_t - train_t, 2), 's')
    # calculate the real accuracy score of this prediction
    train_score = metrics.accuracy_score(y_train, y_pred1)
    test_score = metrics.accuracy_score(y_test, y_pred2)
    # return the best parameter K, the train accuracy score,
    # the mean Cross Validation score, and the test score
    return knn_gscv.best_params_, train_score, \
           knn_gscv.best_score_, test_score

