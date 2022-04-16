### Section 1: Simple Dataset Preparation
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
### Example 1: Classification using DecisionTreeClassifier

### import the sklearn module 'sklearn.tree' that includes DecisionTreeClassifier algorithms
from sklearn.tree import DecisionTreeClassifier

### create the classifier
clf = DecisionTreeClassifier()

### fit (train) the classifier on the training features and labels of subset1
clf.fit(features_subset1, labels_subset1) 

### test the classifier by predicting the labels for features_subset2 
print('\nExample 1 Output ')
print('Prediction result (predicted labels) using DecisionTreeClassifier are: ')
print(clf.predict(features_subset2))  # this will print the predicted labels for subset2 features

### find the prediction accuracy of the classifier  
print('Prediction accuracy using DecisionTreeClassifier is: ')
print(clf.score(features_subset2, labels_subset2)) # this will print the percentage of the correctly classified samples of subset2. Correctly classified samples are the ones that their predicted labels match their true labels

######################################################################
### Example 2: Classification using GaussianNB Classifier

### import the sklearn module 'sklearn.naive_bayes' that includes GaussianNB Classifier algorithms
from sklearn.naive_bayes import GaussianNB 

### create the classifier
clf = GaussianNB()

### fit (train) the classifier on the training features and labels of subset1
clf.fit(features_subset1, labels_subset1) 

### test the classifier by predicting the labels for features_subset2 
print('\nExample 2 Output ')
print('Prediction result (predicted labels) using GaussianNB are: ')
print(clf.predict(features_subset2)) # this will print the predicted labels for subset2 features

### find the prediction accuracy of the classifier  
print('Prediction accuracy using GaussianNB is: ')
print(clf.score(features_subset2, labels_subset2)) # this will print the percentage of the correctly classified samples of subset2. Correctly classified samples are the ones that their predicted labels match their true labels

######################################################################
### Example 3: Classification using SVC Classifier

### import the sklearn module 'sklearn.svm' that includes SVC Classifier algorithms
from sklearn.svm import SVC

### create the classifier
clf = SVC()

### fit (train) the classifier on the training features and labels of subset1
clf.fit(features_subset1, labels_subset1) 

### test the classifier by predicting the labels for features_subset2 
print('\nExample 3 Output ')
print('Prediction result (predicted labels) using SVC are: ')
print(clf.predict(features_subset2))  # this will print the predicted labels for subset2 features

### find the prediction accuracy of the classifier  
print('Prediction accuracy using SVC is: ')
print(clf.score(features_subset2, labels_subset2)) # this will print the percentage of the correctly classified samples of subset2. Correctly classified samples are the ones that their predicted labels match their true labels

######################################################################
### Example 4: Performing random splits on arrays

### import the sklearn module 'sklearn.model_selection' that includes train_test_split
from sklearn.model_selection import train_test_split

### split arrays or matrices into random train and test subsets. The parameters train_size and test_size represent the proportion of the dataset to include in the train subset and in the test subset.
features_subset1, features_subset2, labels_subset1, labels_subset2 = train_test_split(features, labels, train_size=0.50, test_size=0.50) # this will split the data into two equal subsets of training and testing (each subset will have half(50%) of the total number of samples)

### print the subsets for visualization purposes
print('\nExample 4 Output ')
print('Data after random split')
print('Features for subset1 are:\n', features_subset1)
print('Features for subset2 are:\n', features_subset2)
print('Labels for subset1 are:\n', labels_subset1)
print('Labels for subset2 are:\n', labels_subset2)

######################################################################
