# Student Number: 18051000
# This file builds a MLP deep learning model for image classification.
# Hyper-parameter tuning is used to find the optimal output units,
# learning rate for the optimiser, and the number of epochs.
# The obtained hyper-parameters will be stored in a project called
# "untitled_project" in the current directory.
# Part of the code in this file refers to the Tensorflow online tutorial:
# https://www.tensorflow.org/tutorials/keras/classification


import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import time
import numpy as np
import matplotlib.pyplot as plt


def mlp(image_size=128, num_classes=4):
    model = tf.keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(image_size, image_size)))
    model.add(keras.layers.Dense(672, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(672, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


# Build an MLP model for hyper-parameter tuning
# The MLP model consists of three dense-connected layers with two
# regularisation layers.
def mlp_model(hp, image_size=128, num_classes=4):
    model = tf.keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(image_size, image_size)))

    # tune the number of output units in the two dense-connected layers
    hp_units = hp.Int('units', min_value=32, max_value=1024, step=32)
    model.add(keras.layers.Dense(hp_units, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(hp_units, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    # the number of final output should be equal to the number of classes
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # tune the learning rate for the optimiser
    # choosing from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


# Create a tuner for hyper-parameter tuning on MLP model
# Return the created tuner and optimal hyper-parameters
def build_tuner(x_train, y_train, x_test, y_test):
    tuner = kt.Hyperband(mlp_model, objective='val_accuracy',
                         max_epochs=20, factor=3)
    # stop the tuning if there is no improvement on the prediction
    # accuracy after running epochs of "patience" (5 in here)
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # Search for the best hyper-parameters
    tuner.search(x_train, y_train, epochs=50,
                 validation_data=(x_test, y_test), callbacks=[stop_early])
    # get the optimal hyper-parameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print('The optimal number of units in the '
          'two densely-connected layers are: ',
          best_hps.get('units'))
    print('The optimal learning rate for the optimiser is: ',
          best_hps.get('learning_rate'))
    # build and fit the best MLP model with optimal hyper-parameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train, y_train, epochs=50,
                        validation_data=(x_test, y_test))
    print('Initial fitting is complete, epochs: ', 50)
    val_accuracy = history.history['val_accuracy']
    # obtain the number of epochs that brings the highest accuracy
    best_epochs = val_accuracy.index(max(val_accuracy)) + 1
    return tuner, best_hps, best_epochs


# Train and evaluate the MLP model with optimal parameters
def mlp_classification(x_train, y_train, x_test, y_test):
    start_t = time.time()
    tuner, best_hps, best_epochs = \
        build_tuner(x_train, y_train, x_test, y_test)
    # retrain the MLP model with the best parameters and number of epochs
    mlp = tuner.hypermodel.build(best_hps)
    scores = mlp.fit(x_train, y_train, epochs=best_epochs,
                     validation_data=(x_test, y_test))
    print('Final fitting is complete, epochs: ', best_epochs)
    train_t = time.time()
    print('Time consumed for MLP training:',
          round(train_t - start_t, 2), 's')
    evaluation_score = mlp.evaluate(x_test, y_test)
    pred_t = time.time()
    print('Time consumed for MLP evaluation:',
          round(pred_t - train_t, 2), 's')
    predictions = mlp.predict(x_test)
    return scores, evaluation_score, predictions


# Plot a selected image from MRI dataset with predictions from the MLP model
# X-label is blue if the prediction is correct and red if it is wrong
def plot_image(predictions_array, image, true_label):
    classes = ['no', 'glioma', 'meningioma', 'pituitary']
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label],
                                         100 * np.max(predictions_array),
                                         classes[true_label]), color=color)


# Plot the label of an image predicted by the MLP model
# The predicted (wrong) label is in red while the true label is in blue
def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    bar_plot = plt.bar(np.arange(len(predictions_array)),
                       predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    bar_plot[predicted_label].set_color('red')
    bar_plot[true_label].set_color('blue')


# Plot images with their predicted and true labels
def show_images(predictions, test_images, test_labels, rows, cols):
    num_images = rows * cols
    plt.figure(figsize=(2 * 2 * cols, 2 * rows))
    for i in range(num_images):
        plt.subplot(rows, 2 * cols, 2 * i + 1)
        plot_image(predictions[i], test_images[i], test_labels[i])
        plt.subplot(rows, 2 * cols, 2 * i + 2)
        plot_value_array(predictions[i], test_labels[i])
    plt.tight_layout()
    plt.show()
