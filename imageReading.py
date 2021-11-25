# Student Number: 18051000
# This file aims at getting features of images in the MRI dataset
# Images are read with grayscale pixels stored as their features


import os
import numpy as np
import pandas as pd
import cv2
from skimage.io import imread
from skimage.transform import resize


global basedir, image_paths, target_size
basedir = 'D:\Study\MLNotebook\AMLS_21-22_SN18051000'
images_dir = os.path.join(basedir, 'image')  # directory of images
labels_filename = 'label.csv'  # directory of labels


# image_reading() reads images and sets their labels to corresponding integers
def image_reading():
    labels_df = pd.read_csv(os.path.join(basedir, labels_filename))
    image_names = labels_df.iloc[:3000, 0]
    features = []
    for image_path in image_names:
        # 512x512 pixels are too large to store in an array
        # skimage provides a resize function to reduce the image resolution
        image = imread(os.path.join(images_dir, image_path), as_gray=True)
        image = resize(image, (128, 128))  # 128x128 image resolution
        feature = np.reshape(image, (128*128))
        features.append(np.array(feature))

    label_names = labels_df.iloc[:3000, 1]
    labels = []
    for tumor in label_names:
        if tumor == 'no_tumor':
            labels.append(0)
        # for binary classification
        else:
            labels.append(1)
        # for multiclass classification
        # elif tumor == 'glioma_tumor':
        #     labels.append(1)
        # elif tumor == 'meningioma_tumor':
        #     labels.append(2)
        # else:
        #     labels.append(3)

    # return features of all the images and their labels
    return features, labels


# feature_extraction() extracts FAST, BREIF or ORB features in each image
# def feature_extraction(image_path):
#     image = cv2.imread(os.path.join(images_dir, image_path), 0)
#     orb = cv2.ORB_create()
#     key_points = orb.detect(image, None)
#     kp_list = np.zeros(2 * len(key_points))
#     idx = 0
#     for kp in key_points:
#         kp_list[idx] = kp.pt[0]
#         kp_list[idx + 1] = kp.pt[1]
#         idx += 2
#     return kp_list