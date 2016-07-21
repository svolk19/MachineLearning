import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import time
import neural_net_regression as neural_net


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

    startTime = time.time()
    NN.gradient_adjust(X_train, y_train, iterations=1000, learning_rate=0.001, regularize=True)
    endTime = time.time()

    totalTime = endTime - startTime

    accuracy = NN.accuracy(X_test, y_test)
    print(accuracy, 'total time:', totalTime)


def test(NN):

    # teaches neural net correlation between hours studied, hours slept and test score

    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    X = X/np.amax(X)
    y = y/100

    NN.train(X, y, learning_rate=0.001, iterations=100, regularize=False
             )

    return NN.accuracy(X, y, string=True)


def Xor(NN):
    # teach neural net Xor function
    X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

    NN.train(X, y, learning_rate=0.001, iterations=10000, regularize=True)
    return NN.accuracy(X, y, string=True)


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

    startTime = time.time()
    NN.train(X_train, y_train, iterations=10000, learning_rate=0.001, regularize=False, reg_lambda=0.01, print_accuracies=True)
    endTime = time.time()


    totalTime = endTime - startTime
    print('accuracy:' + str(NN.accuracy(X_test, y_test)) + '%\n' + 'total time: ' + str(totalTime))

if __name__ == '__main__':
    print('--------------------------\n' + 'process started\n' + '--------------------------\n')
    NN = neural_net.NeuralNetwork(13, 1, 10, 10)
    boston_housing(NN)