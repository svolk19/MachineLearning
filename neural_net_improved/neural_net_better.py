import numpy as np
import math
import matplotlib.pyplot as plt

class neural_network(object):
    def __init__(self):
        #layer sizes
        self.inputSize = 2
        self.outputSize = 1
        self.hidden1Size = 5
        self.hidden2Size = 5

        #initialize random primary weight scheme
        self.w1 = np.random.randn(self.inputSize, self.hidden1Size)
        self.w2 = np.random.randn(self.hidden1Size, self.hidden2Size)
        self.w3 = np.random.randn(self.hidden2Size, self.outputSize)

    def sigmoid(self, X, deriv=False):
        # activation function at each neuron: tanh
        # assumes X is a numpy array
        if deriv:
            return 1.0 - X ** 2
        else:
            for i, column in enumerate(X):
                for j, elem in enumerate(column):
                    X[i][j] = math.tanh(elem)
            return X

    def predict(self, X):
        # feed forward dataset X
        # assumes X is a numpy array, returns numpy array
        self.z2 = np.dot(X, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.w3)
        self.a4 = self.sigmoid(self.z4)
        return self.a4

    def backPropagate(self, X, y):
        # backpropagation of value X
        yhat = self.predict(X)

        delta4 = np.multiply(-(y - yhat),
                             self.sigmoid(self.z4, deriv=True))
        dJdW3 = np.dot(self.a3.T, delta4)
        delta3 = np.dot(delta4, self.w3.T) * self.sigmoid(self.z3, deriv=True)
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.w2.T) * self.sigmoid(self.z2, deriv=True)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2, dJdW3

    def gradient_adjust(self, X, y, iterations=1000, learning_rate=0.5, display=False):

        # train neural network until greater than or equal to 99.5% accuracy is achieved
        for num in range(iterations):

            # calculate gradients
            dJdw1, dJdw2, dJdw3 = self.backPropagate(X, y)

            # train w1, w2, and w3
            for i, deriv in enumerate(dJdw1):
                self.w1[i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdw2):
                self.w2[i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdw3):
                self.w3[i] -= learning_rate * deriv

            if display:
                plt.scatter(num, self.accuracy(X, y))

        if display:
            plt.show()

    def train(self, X, y, iterations=1000, learning_rate=0.5, display=False, reinitialize=False):
        # neural net training

        numPasses = 0
        learnRate = learning_rate
        testAccuracy = 0

        while testAccuracy <= 0.95 and numPasses <= 15:

            numPasses += 1

            if numPasses % 3 == 0:
                learnRate += 0.1
                self.gradient_adjust(X, y, iterations=iterations, learning_rate=learnRate, display=display)

            self.gradient_adjust(X, y, iterations=iterations, learning_rate=learning_rate, display=display)

            testAccuracy = self.accuracy(X, y)
            print(testAccuracy)

        if self.accuracy(X, y) < 0.95 and reinitialize:
            # try a different random weighting
            self.w1 = np.random.randn(self.inputSize, self.hidden1Size)
            self.w2 = np.random.randn(self.hidden1Size, self.hidden2Size)
            self.w3 = np.random.randn(self.hidden2Size, self.outputSize)

            self.train(X, y)

        return 'accuracy:' + str(self.accuracy(X, y))


    def accuracy(self, X, y, string=False):
        # produces the accuracy of neural net
        yhat = self.predict(X)
        errorDifference = np.absolute(np.subtract(yhat, y))
        error = np.divide(errorDifference, y)
        accuracy = 1.0 - np.average(error)

        if string:
            print("accuracy: " + str(accuracy * 100.0) + "%")
        else:
            return accuracy

    def error(self, X, y):
        # produces the total squared error of neural net
        error = 0.0
        yHat = self.predict(X)
        for i in range(len(yHat)):
            error += (y[i] - yHat[i]) ** 2

        return error


def Xor():
    # teach neural net Xor function
    X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

    NN = neural_network()

    print('accuracy before training: ' + str(NN.accuracy(X, y) * 100.0) + '%')

    NN.train(X, y)

    return NN.accuracy(X, y, string=True)



def test():

    # teaches neural net correlation between hours studied, hours slept and test score

    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    X = X/np.amax(X)
    y = y/100

    NN = neural_network()

    print('accuracy before training: ' + str(NN.accuracy(X, y) * 100.0) + '%')

    NN.train(X, y)

    return NN.accuracy(X, y, string=True)

if __name__ == '__main__':
    # print('---------------------------\n\n' + 'test score predictor\n\n' + '---------------------------\n\n')
    # test()
    #
    # print('---------------------------\n\n' + 'Xor calculator\n\n' + '---------------------------\n\n')
    # Xor()
    #
    #

    test()
