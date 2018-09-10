# Initialize classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
clf_naive_bayes = GaussianNB()

from sklearn.ensemble import AdaBoostClassifier
clf_adaboost = AdaBoostClassifier(n_estimators=2)

from sklearn import svm
clf_svm = svm.SVC()


# Data
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)


# Train classifier and predict
clf = clf.fit(X_train, Y_train)
prediction_decision_tree = clf.predict(X_test)

clf_svm = clf_svm.fit(X_train, Y_train)
prediction_svm = clf_svm.predict(X_test)

clf_adaboost = clf_adaboost.fit(X_train, Y_train)
prediction_adaboost = clf_adaboost.predict(X_test)

clf_naive_bayes = clf_naive_bayes.fit(X_train, Y_train)
prediction_naive_bayes = clf_naive_bayes.predict(X_test)


# Calculate accuracy
from sklearn.metrics import accuracy_score
acc_dec_tree = accuracy_score(prediction_decision_tree, Y_test)
acc_svm = accuracy_score(prediction_svm, Y_test)
acc_ada = accuracy_score(prediction_adaboost, Y_test)
acc_naive = accuracy_score(prediction_naive_bayes, Y_test)


# Print classifier with highest accuracy
import numpy as np
index = np.argmax([acc_dec_tree, acc_svm, acc_ada, acc_naive])
classifiers = {0:'Decision Tree', 1: 'SVM', 2: 'Adaboost', 3: 'Naive bayes'}
print('Best gender classifier is: {}'.format(classifiers[index]))
