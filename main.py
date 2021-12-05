# Student Number: 18051000
# This file contains the main function to run this program.
# The program is able to classify a group of MRI images using three
# different machine learning models: KNN, SVM, MLP.
# Steps to run the program:
# 1. Run the main function and two questions will pop up.
# 2. Type to answer the first question by choosing a machine learning model.
# 3. Type to answer the second question by choosing the classification task.
# 4. Wait to see how successful the model is (the prediction accuracy).
# For KNN and SVM, the prediction accuracy can be visualised as numbers.
# For MLP, the prediction accuracy can be visualised as numbers, line charts,
# and grid images with histograms.


import imageReading
import knnClassifier
import svmClassifier
import mlpClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


if __name__ == '__main__':
    model_names = {'KNN', 'SVM', 'MLP'}
    task_names = {'binary', 'multiclass'}
    # Keep asking until correct model and task are selected
    model = input('Please choose the name of model for classification '
                  '(KNN, SVM, MLP): ')
    while model not in model_names:
        model = input('Please choose the name of model for classification '
                      '(KNN, SVM, MLP): ')
    task = input('Please choose the classification task '
                 '(binary, multiclass): ')
    while task not in task_names:
        task = input('Please choose the classification task '
                     '(binary, multiclass): ')

    # Get image features and labels as a dataset
    start_t = time.time()
    X_features, Y_labels = imageReading.image_reading(model, task)
    print('Time consumed for image reading:',
          round(time.time() - start_t, 2), 's')
    # Split 3000 images and labels into training and testing datasets
    # random_state is set to 0 so the dataset will not be randomly divided
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X_features, Y_labels, train_size=0.7, random_state=0)
    # imageReading.show_images(X_test, Y_test)

    if model == 'KNN':
        # KNN training and validating
        best_k, train_score, best_score, validate_score = \
            knnClassifier.knn_classification(X_train, Y_train, X_test, Y_test)
        print('The optimal parameter being used is: ', best_k)
        print('KNN Training error: ', 1 - train_score)
        print('Validation score from Cross Validation: ', best_score)
        print('Validation error from Cross Validation: ', 1 - best_score)
        print('Validation score from the best model: ', validate_score)
        print('Validation error from the best model: ', 1 - validate_score)

    elif model == 'SVM':
        # SVM training and validating
        best_params, train_score, best_score, validate_score = \
            svmClassifier.svm_classification(X_train, Y_train, X_test, Y_test)
        print('The optimal parameters being used are: ', best_params)
        print('SVM Training error: ', 1 - train_score)
        print('Validation score from Cross Validation: ', best_score)
        print('Validation error from Cross Validation: ', 1 - best_score)
        print('Validation score from the best model: ', validate_score)
        print('Validation error from the best model: ', 1 - validate_score)

    elif model == 'MLP':
        # MLP training and validating
        # mlp = mlp()
        # mlp.fit(X_train, Y_train, epochs=50,
        #                     validation_data=(X_test, Y_test))
        # evaluation_score = mlp.evaluate(X_test, Y_test)
        # print('Validation score from the best model: ', evaluation_score)

        scores, evaluation_result, predictions = \
            mlpClassifier.mlp_classification(X_train, Y_train, X_test, Y_test)

        # Display the prediction accuracy of first 15 images
        mlpClassifier.show_images(predictions, X_test, Y_test, 5, 3)

        # Plot epochs versus training and validation accuracy of the MLP model
        plt.plot(scores.history['accuracy'], label='accuracy')
        plt.plot(scores.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        print('Validation score from the best model: ', evaluation_result[1])
        print('Validation error from the best model: ', 1 - evaluation_result[1])

    else:
        print('Wrong model name!')

