import neural_net_classifier as neural_net
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import numpy as np


def iris(NN):
    # iris test data classification problem from sklearn

    data = datasets.load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

    X_train.reshape((105, 4))
    X_test.reshape((45, 4))

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

    y_trainList = []
    for i, elem in enumerate(y_train):
        if elem == 0:
            y_trainList.append([1, 0, 0])
            continue
        if elem == 1:
            y_trainList.append([0, 1, 0])
            continue
        else:
            y_trainList.append([0, 0, 1])
            continue
    y_train = np.array(y_trainList)
    y_train.reshape((105, 3))

    y_testList = []
    for i, elem in enumerate(y_test):
        if elem == 0:
            y_testList.append([1, 0, 0])
            continue
        if elem == 1:
            y_testList.append([0, 1, 0])
            continue
        else:
            y_testList.append([0, 0, 1])
            continue
    y_test = np.array(y_testList)
    y_test.reshape((45, 3))

<<<<<<< HEAD
    NN.train(X_train, y_train, learning_rate=0.001, iterations=1000, regularize=False, regChange=0.01)
=======
    NN.train(X_train, y_train, learning_rate=0.001, iterations=100000, regularize=False)
>>>>>>> master
    print(NN.accuracy(X_test, y_test))

if __name__  == '__main__':
    NN = neural_net.neural_network(4, 3, 5, 5)
    iris(NN)