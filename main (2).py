import numpy as np

#Starter code to import iris dataset
from sklearn.datasets import load_iris
features, labels = load_iris(return_X_y = True); 
from sklearn.model_selection import train_test_split


#Solutions to part A

print("\nPart A")


print("\n")
print("\n")
print("\nGaussianNB classifier")
print("------------------------------------------")
from sklearn.naive_bayes import GaussianNB 

totalAccuracy = 0
for i in range(10):
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.7, test_size=0.3)

  classifier = GaussianNB()
  classifier.fit(features_train, labels_train)
  classifier.predict(features_test)
  print("\nPrediction accuracy for GaussianNB Classifier:", classifier.score(features_test, labels_test))
  totalAccuracy = totalAccuracy + classifier.score(features_test, labels_test)
GaussianNB_Avg_Accuracy = totalAccuracy/10
print("Average Prediction Accuracy for GaussianNB:", (GaussianNB_Avg_Accuracy))




#Solutions to part B
print("\n")
print("\n")
print("\n")
print("\n")
print("\nPart B")
print("------------------------------------------")
from sklearn.neighbors import KNeighborsClassifier
totalAccuracy = 0
for i in range(10):
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.7, test_size=0.3)

  classifier = KNeighborsClassifier()
  classifier.fit(features_train, labels_train)
  classifier.predict(features_test)
  print("\nPrediction accuracy for KNeighbors Classifier:", classifier.score(features_test, labels_test))
  totalAccuracy = totalAccuracy + classifier.score(features_test, labels_test)
KNeighborsClassifier_Avg_Accuracy = totalAccuracy / 10
print("Average Prediction Accuracy for KNeighbors Classifier:", (KNeighborsClassifier_Avg_Accuracy))


print("\n")
print("\n")
print("\n")
print("\n")
print("\nPart C")
print("\n")

print("\nSupport Vector Machines Classifier")
print("------------------------------------------")
from sklearn.svm import SVC
totalAccuracy = 0
for i in range(10):
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.7, test_size=0.3)

  classifier = SVC()
  classifier.fit(features_train, labels_train)
  classifier.predict(features_test)
  print("\nPrediction accuracy:", classifier.score(features_test, labels_test))
  totalAccuracy = totalAccuracy + classifier.score(features_test, labels_test)
SVM_Avg_Accuracy = totalAccuracy / 10
print("Average Prediction Accuracy for SVM classifier:", (SVM_Avg_Accuracy))  


print("\n")
print("\n")
print("\n")
print("\n")
print("\nDecisionTreeClassifier")
print("------------------------------------------")
from sklearn.tree import DecisionTreeClassifier
totalAccuracy = 0
for i in range(10):
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.7, test_size=0.3)

  classifier = DecisionTreeClassifier()
  classifier.fit(features_train, labels_train)
  classifier.predict(features_test)
  print("\nPrediction accuracy:", classifier.score(features_test, labels_test))
  totalAccuracy = totalAccuracy + classifier.score(features_test, labels_test)
DecisionTreeClassifier_Avg_Accuracy = totalAccuracy / 10
print("Average Prediction Accuracy for DecisionTreeClassifier:", (DecisionTreeClassifier_Avg_Accuracy))  




print("\n")
print("\n")
print("\n")
print("\n")
print("\nAdaBoostClassifier")
print("------------------------------------------")
from sklearn.ensemble import AdaBoostClassifier
totalAccuracy = 0
for i in range(10):
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.7, test_size=0.3)

  classifier = AdaBoostClassifier()
  classifier.fit(features_train, labels_train)
  classifier.predict(features_test)
  print("\nPrediction accuracy:", classifier.score(features_test, labels_test))
  totalAccuracy = totalAccuracy + classifier.score(features_test, labels_test)
AdaBoostClassifier_Avg_Accuracy = totalAccuracy / 10
print("Average Prediction Accuracy for AdaBoostClassifier:", (AdaBoostClassifier_Avg_Accuracy))  


print("\nPart D")
accuracies = np.array([GaussianNB_Avg_Accuracy, KNeighborsClassifier_Avg_Accuracy, SVM_Avg_Accuracy, DecisionTreeClassifier_Avg_Accuracy, AdaBoostClassifier_Avg_Accuracy])
highestAccuracy = np.max(accuracies)

