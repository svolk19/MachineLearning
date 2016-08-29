import neural_net.classifier_NN.neural_net_classifier as neural_net
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

    NN.train(X_train, y_train, learning_rate=0.001, iterations=1000, regularize=False, display=True)
    print(NN.accuracy(X_test, y_test))


def digits(NN):
    # hand written digit test data classification problem from sklearn

    data = datasets.load_digits(n_class=10)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

    X_train.reshape((1257, 64))
    X_test.reshape((540, 64))

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

    y_train_class_array = np.zeros((len(y_train), 10))
    for i, class_label in enumerate(y_train):
        y_train_class_array[i][class_label] = 1

    y_test_class_array = np.zeros((len(y_test), 10))
    for i, class_label in enumerate(y_test):
        y_test_class_array[i][class_label] = 1

    y_train = y_train_class_array.reshape((1257, 10))
    y_test = y_test_class_array.reshape((540, 10))

    NN.train(X_train, y_train, learning_rate=0.01, iterations=100, regularize=False, display=True)
    print(NN.accuracy(X_test, y_test))


if __name__ == '__main__':
    print('----------------\n' + 'process started\n' + '----------------\n')

    NN = neural_net.NeuralNetwork(64, 10, 15, 15)
    digits(NN)
