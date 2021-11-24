# Student Number: 18051000
# This is an initial trial of KNN Classifier with the iris dataset
# The code is written based on the first exercise of Week 4


import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def knn_classifier(x_train, y_train, x_test, k):
    # Create a KNN classifier with k neighbours
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train) # Fit KNN model
    y_pred = neigh.predict(x_test)
    return y_pred


if __name__ == '__main__':
    irisData = load_iris()  # get the example dataset
    print(irisData.data.shape)
    print(irisData.feature_names)
    # convert the dataset into a dataframe
    irisData_df = pd.DataFrame(irisData.data, columns=irisData.feature_names)
    irisData_df['Species'] = irisData.target
    newX = irisData_df.drop('Species', axis=1)  # 4 columns of features
    newY = irisData_df['Species']  # one column of target

    # split the dataset into two: training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(newX, newY, test_size=0.3, random_state=3)

    # predict the species of the testing dataset and calculate the accuracy score
    Y_pred = knn_classifier(X_train, Y_train, X_test, 5)
    score = metrics.accuracy_score(Y_test, Y_pred)
    print(score)

