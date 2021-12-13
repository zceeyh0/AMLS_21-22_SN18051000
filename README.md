# AMLS_21-22_SN18051000
* This is a repository for the final assignment of ELEC0134, created by student 18051000. <br>
* The program is able to build three machine learning models to implement image classification tasks. <br>
* The dataset being used is the Brain Tumor Classification (MRI) dataset. <br>
* Three models created by this program are KNN, SVM, MLP. <br>
* The whole program is written in Python. <br>

## Main tasks that are achieved:
#### 1. Binary classification task: <br>
   Building machine learning models to identify whether there is a tumor in each MRI image. <br>
#### 2. Multiclass classification task: <br>
   Building machine learning models to identify the type of tumor in each MRI image (no tumor, glioma tumor, meningioma tumor, or pituitary tumor). <br>

## Organisation of files and their roles:
This repository consists of five python files, a README.md file and two folders (MRI dataset for training and testing): <br>
#### 1. `README.md`: <br>
Introduction to this repository with instructions on how to compile and run the code. <br>
#### 2. Folders `train` and `test`:  <br>
The folder `train` includes 3000 MRI images for training and validation. The folder `test` includes 200 MRI images for evaluating the accuracy of models. The two csv files `label.csv` in both folders contain filenames of all the MRI images and their corresponding labels (types of tumor). <br>
#### 3. `imageReading.py`: <br>
This file gets features (pixel values) of images in the MRI dataset and their labels.
#### 4. `knnClassifier.py`: <br>
This file builds a KNN (non-deep learning) Classifier for classification tasks.
#### 5. `svmClassifier.py`: <br>
This file builds a SVM (non-deep learning) Classifier for classification tasks.
#### 6. `mlpClassifier.py`: <br>
This file builds a MLP deep learning model for classification tasks.
#### 7. `main.py`: <br>
This file contains the `main` function to run this program.

## The necessary Python packages/modules are: <br>
`numpy 1.20.3`, `pandas 1.3.4`, `scikit-learn 1.0.1`, `scikit-image 0.18.3`, `matplotlib 3.5.0`, `tensorflow 2.7.0`, `keras 2.7.0`, `keras_tuner 1.1.0` <br>
When downloading these packages/modules, other essential packages/modules will be downloaded automatically as dependencies.

## To run the code, please follow the steps below: <br>
1. Download the whole repository (either by `git clone` or download zip). <br>
2. Make sure you have installed all the necessary Python packages/modules (listed above). <br>
3. Open `main.py` and run the main function. Two questions will pop up. <br>
4. Type a number (from 1 to 3) to choose a machine learning model (KNN, SVM, MLP). <br>
5. Type a number (1 or 2) to choose a classification task (binary, multiclass). <br>
6. Wait to see how successful your model is by checking the prediction accuracy displayed in different forms (scores in percentages, line charts, grid images with histograms). <br>
#### Please Note: You may see a folder called 'untitled_project' generated in the project directory when you run the MLP model. This is the record of parameters from hyper-parameter tuning, which avoids repeating the tuning process and can be deleted once you don't need the parameters anymore. <br>
