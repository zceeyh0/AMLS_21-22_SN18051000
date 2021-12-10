# Student Number: 18051000
# This file gets features of images in the MRI dataset and their labels.
# Images are read and resized to 128x128 grayscale pixels.
# Labels are set between 0 and 1 for binary classification, and from 0 to
# 3 for multiclass classification.
# After reading all images, show_image() can display images (16 at most).


import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize


global basedir, image_paths, target_size
basedir = 'D:\Study\MLNotebook\AMLS_21-22_SN18051000'
images_dir = os.path.join(basedir, 'image')  # directory of images
labels_filename = 'label.csv'  # directory of labels
class_names = ['no_tumor', 'glioma_tumor',
               'meningioma_tumor', 'pituitary_tumor']


# Read images and return a dataset with images and converted image labels
def image_reading(model, task):
    labels_df = pd.read_csv(os.path.join(basedir, labels_filename))
    image_names = labels_df.iloc[:3000, 0]  # 3000 images
    features = []
    for image_path in image_names:
        image = imread(os.path.join(images_dir, image_path), as_gray=True)
        # 512x512 pixels are too large to store in a normal numpy array
        # skimage provides a resize function to reduce the image resolution
        feature = resize(image, (128, 128))  # 128x128 image resolution
        # Reshape image features to 1-dimension arrays for supervised learning
        if model == 'KNN' or model == 'SVM':
            feature = np.reshape(feature, (128*128))
        features.append(np.array(feature / 255.0))  # normalization

    label_names = labels_df.iloc[:3000, 1]  # 3000 labels
    labels = []
    # Store labels for binary classification
    # 0 represents no tumor, while 1 stands for having a tumor
    if task == 'binary':
        for tumor in label_names:
            if tumor == 'no_tumor':
                labels.append(0)
            else:
                labels.append(1)
    # Store labels for multiclass classification
    # 0 represents no tumor, 1 stands for having a glioma_tumor,
    # 2 means having a meningioma_tumor, 3 means having a pituitary tumor
    elif task == 'multiclass':
        for tumor in label_names:
            if tumor == 'no_tumor':
                labels.append(0)
            elif tumor == 'glioma_tumor':
                labels.append(1)
            elif tumor == 'meningioma_tumor':
                labels.append(2)
            else:
                labels.append(3)
    else:
        print("Wrong task name!")
        return 0

    # return features of all the images and their labels
    return np.asarray(features), np.asarray(labels)


# Display first 16 images with their labels
def show_images(images, labels, num=16):
    plt.figure(figsize=(10, 10))
    size = int(math.sqrt(num))
    for i in range(num):
        plt.subplot(size, size, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.reshape(images[i], (128, 128)), cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

