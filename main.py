# Student Number: 18051000
# This file contains the main function to run my program
# A binary/multiclass classifier is built to identify labels of images
# Models: KNN Classifier, ...


import imageReading
import knnClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == '__main__':
    # get image features and labels as a dataset
    X_features, Y_labels = imageReading.image_reading()

    # find the best parameter k for training this dataset
    # best_k, best_score = knnClassifier.hypertuning(X_features, Y_labels, 5)
    # print(best_k, best_score)

    # train and test the dataset directly using KNN Classifier
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X_features, Y_labels, train_size=0.7)
    Y_pred = knnClassifier.knn_classifier(X_train, Y_train, X_test, 3)
    score = metrics.accuracy_score(Y_test, Y_pred)
    print(score)

    # train and test the dataset multiple times using Cross Validation
    cv_scores = knnClassifier.cross_validation(X_features, Y_labels, 3, 5)
    print(cv_scores)
    print('cv_scores mean: {}'.format(np.mean(cv_scores)))


