import neural_net.regression_NN.neural_net_regression as neural_net
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split
import time


def boston_housing(NN):
    data = datasets.load_boston()

    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    X_train = np.array(X_train).reshape((404, 13))
    X_test = np.array(X_test).reshape((102, 13))
    y_train = np.array(y_train).reshape((404, 1))
    y_test = np.array(y_test).reshape((102, 1))

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)
    y_train = preprocessing.normalize(y_train)
    y_test = preprocessing.normalize(y_test)

    start_time = time.time()
    NN.train(X_train, y_train, iterations=100, learning_rate=0.01, regularize=False, display=True)
    end_time = time.time()

    total_time = end_time - start_time
    print(NN.accuracy(X_test, y_test), 'total time:', total_time)


def diabetes_test(NN):
    data = datasets.load_diabetes()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    X_train = np.array(X_train).reshape((353, 10))
    X_test = np.array(X_test).reshape((89, 10))
    y_train = np.array(y_train).reshape((353, 1))
    y_test = np.array(y_test).reshape((89, 1))

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)
    y_train = preprocessing.normalize(y_train)
    y_test = preprocessing.normalize(y_test)

    start_time = time.time()
    NN.train(X_train, y_train, iterations=100, learning_rate=0.001, regularize=False, reg_lambda=0.01, display=True)
    end_time = time.time()


    total_time = end_time - start_time
    print('accuracy:' + str(NN.accuracy(X_test, y_test)) + '%\n' + 'total time: ' + str(total_time))


def xor(NN):
    # teach neural net Xor function
    X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

    NN.train(X, y)
    return NN.accuracy(X, y, string=True)

if __name__ == '__main__':
    NN = neural_net.NeuralNetwork(13, 1, 10, 10)
    boston_housing(NN)
