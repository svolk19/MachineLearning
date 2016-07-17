from sklearn import datasets
import numpy as np
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

from sklearn.neighbors import KNeighborsClassifier

my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
predictions = np.array(predictions)
y_test = np.array(y_test)

print(predictions, y_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))


