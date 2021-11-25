# Student Number: 18051000
# This file aims at building KNN Classifier for training and testing datasets
# A KNN Classifier is built to train and test data directly
# Cross Validation method is used to train models in multiple groups
# Grid Search hypertuning method is used to find the optimal parameter


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# knn_classifier() builds a KNN Classifier, trains and tests the input datasets
def knn_classifier(x_train, y_train, x_test, k):
    # create a KNN classifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train) # fit KNN model
    y_pred = knn.predict(x_test) # predict on the test dataset
    return y_pred


# cross_validation() trains and tests multiple groups of data to build
# the most accurate model
def cross_validation(x, y, k, cv):
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    # cv: number of groups divided from the dataset for training
    cv_scores = cross_val_score(knn_cv, x, y, cv=cv)
    # return a list of scores for all the training groups
    return cv_scores


# hypertuning() finds the best parameter for k
def hypertuning(x, y, cv):
    knn_tuning = KNeighborsClassifier()
    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, 50)}
    # use Grid Search to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn_tuning, param_grid, cv=cv)
    # fit model to data
    knn_gscv.fit(x, y)
    # return the k parameter and the highest accuracy score
    return knn_gscv.best_params_, knn_gscv.best_score_
