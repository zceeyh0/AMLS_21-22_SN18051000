# Student Number: 18051000
# This file contains the main function to run my program
# A binary/multiclass classifier is built to identify labels of MRI images
# Models: KNN Classifier, SVM Classifier


import imageReading
import knnClassifier
import svmClassifier
import time
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # get image features and labels as a dataset
    start_t = time.time()
    X_features, Y_labels = imageReading.image_reading()
    print('Time consumed for image reading:',
          round(time.time() - start_t, 2), 's')
    # split 3000 images and labels into training and testing datasets
    # random_state is not set to 0 so the dataset will be randomly divided
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X_features, Y_labels, train_size=0.7)

    model = input('Please choose the name of model for classification '
                  '(knn, svm): ')
    if model == 'knn':
        # KNN training and validating
        best_k, train_score, best_score, validate_score = \
            knnClassifier.knn_classifier(X_train, Y_train, X_test, Y_test, 5)
        print('The optimal parameter being used is: ', best_k)
        print('KNN Training error: ', 1 - train_score)
        print('Validation error from Cross Validation: ', 1 - best_score)
        print('Validation error from the best model: ', 1 - validate_score)
    elif model == 'svm':
        # SVM training and validating
        best_params, train_score, best_score, validate_score = \
            svmClassifier.svm_classifier(X_train, Y_train, X_test, Y_test, 5)
        print('The optimal parameters being used are: ', best_params)
        print('SVM Training error: ', 1 - train_score)
        print('Validation error from Cross Validation: ', 1 - best_score)
        print('Validation error from the best model: ', 1 - validate_score)
    else:
        print('Wrong model name!')

