from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

### Using Section 1: Simple Dataset Preparation
import numpy as np

### Create a dataset of 2D points as samples. The features are the X,Y coordinates of the points and the labels are class 1 or class 2 
features = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
labels = np.array([1, 1, 1, 2, 2, 2])

### Split the dataset manually into two subsets
features_subset1 = [features[0],features[2],features[3]]
labels_subset1 = [labels[0], labels[2], labels[3]]

features_subset2 = [features[1],features[4],features[5]]
labels_subset2 = [labels[1], labels[4], labels[5]]

######################################################################
### Sample Q1 Solution

### import the sklearn module 'sklearn.tree' that includes DecisionTreeClassifier algorithms
from sklearn.tree import DecisionTreeClassifier

### create the classifier
clf = DecisionTreeClassifier()

### fit (train) the classifier on the training features and labels of subset2
clf.fit(features_subset2, labels_subset2) 

### test the classifier by predicting the labels for features_subset1 
print('\nSample Q1 Solution Output')
print('Prediction result (predicted labels) for features_subset1 using DecisionTreeClassifier are: ')
print(clf.predict(features_subset1))  # this will print the predicted labels for subset1 features

### find the prediction accuracy of the classifier  
print('Prediction accuracy using DecisionTreeClassifier on subset 1 is: ')
print(clf.score(features_subset1, labels_subset1)) # this will print the percentage of the correctly classified samples of subset1. Correctly classified samples are the ones that their predicted labels match their true labels

######################################################################
### Sample Q2 Solution

# create a new array names newTestArray
newTestArray = [[-4, -2], [0, 1]]

### import the sklearn module 'sklearn.naive_bayes' that includes GaussianNB Classifier algorithms
from sklearn.naive_bayes import GaussianNB 

### create the classifier
clf = GaussianNB()

### fit (train) the classifier on the training features and labels of subset1
clf.fit(features_subset1, labels_subset1) 

### test the classifier by predicting the labels for newTestArray 
print('\nSample Q2 Solution Output')
print('Prediction result (predicted labels) for newTestArray using GaussianNB are: ')
print(clf.predict(newTestArray)) # this will print the predicted labels for newTestArray features

######################################################################
### Sample Q3 Solution

### import the sklearn module 'sklearn.model_selection' that includes train_test_split
from sklearn.model_selection import train_test_split

### Split arrays or matrices into random train and test subsets. The parameters train_size and test_size represent the proportion of the dataset to include in the train subset and in the test subset.
features_subset1, features_subset2, labels_subset1, labels_subset2 = train_test_split(features, labels, train_size=0.67, test_size=0.33)# this will split the data into two-thirds (67%) for training (subset1) and one-third (33%) for testing (subset2)

### import the sklearn module 'sklearn.svm' that includes SVC Classifier algorithms
from sklearn.svm import SVC

### create the classifier
clf = SVC()

### fit (train) the classifier on the training features and labels of subset1
clf.fit(features_subset1, labels_subset1) 

### test the classifier by predicting the labels for features_subset2
print('\nSample Q3 Solution Output')
print('Prediction result (predicted labels) for features_subset2 using SVC is: ')
print(clf.predict(features_subset2))  # this will print the predicted labels for subset2 features

### Find the prediction accuracy of the classifier  
print('Prediction accuracy using SVC on subset2 is: ')
print(clf.score(features_subset2, labels_subset2)) # this will print the percentage of the correctly classified samples of subset2. Correctly classified samples are the ones that their predicted labels match their true labels

######################################################################


