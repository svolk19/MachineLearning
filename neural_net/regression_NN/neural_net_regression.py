import numpy as np
import matplotlib.pyplot as plt

class neural_network(object):
    def __init__(self, inputSize, outputSize, hidden1Size, hidden2Size):

        # layer sizes
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hidden1Size = hidden1Size
        self.hidden2Size = hidden2Size

        # initialize random primary weight and bias scheme
        self.w1 = np.random.randn(self.inputSize, self.hidden1Size)
        self.w2 = np.random.randn(self.hidden1Size, self.hidden2Size)
        self.w3 = np.random.randn(self.hidden2Size, self.outputSize)
        self.b1 = np.zeros(self.hidden1Size)
        self.b2 = np.zeros(self.hidden2Size)
        self.b3 = np.zeros(self.outputSize)

        # initialize forward propagation parameters to 0.0
        self.z2 = 0.0
        self.a2 = 0.0
        self.z3 = 0.0
        self.a3 = 0.0
        self.z4 = 0.0
        self.a4 = 0.0

    def sigmoid(self, X, deriv=False):
        # activation function at each neuron: tanh
        # assumes X is a numpy array
        if deriv:
            return 1.0 - X ** 2
        else:
            for i, column in enumerate(X):
                for j, elem in enumerate(column):
                    X[i][j] = np.tanh(elem)
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

        # backpropagation of value
        self.yHat = self.predict(X)

        delta4 = np.multiply(-(y - self.yHat),
                             self.sigmoid(self.z4, deriv=True))
        dJdW3 = np.dot(self.a3.T, delta4)
        dJdB3 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = np.dot(delta4, self.w3.T) * self.sigmoid(self.z3, deriv=True)
        dJdW2 = np.dot(self.a2.T, delta3)
        dJdB2 = np.sum(delta3, axis=0)
        delta2 = np.dot(delta3, self.w2.T) * self.sigmoid(self.z2, deriv=True)
        dJdW1 = np.dot(X.T, delta2)
        dJdB1 = np.sum(delta2, axis=0)

        return dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3

    def gradient_adjust(self, X, y, iterations=1000, learning_rate=0.5, regChange=0.01, display=False,
                        regularize=False):

        # train neural network until greater than or equal to 99.5% accuracy is achieved
        for num in range(iterations):

            # calculate gradients
            dJdw1, dJdw2, dJdw3, dJdB1, dJdB2, dJdB3 = self.backPropagate(X, y)

            # train weights and biases

            if regularize:
                self.w1 += regChange * self.w1
                self.w2 += regChange * self.w2
                self.w3 += regChange * self.w3

            for i, deriv in enumerate(dJdw1):
                self.w1[i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdw2):
                self.w2[i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdw3):
                self.w3[i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdB1):
                self.b1[i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdB2):
                self.b2[i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdB3):
                self.b3[i] -= learning_rate * deriv

            if display:
                plt.scatter(num, self.accuracy(X, y))

        if display:
            plt.show()

    def train(self, X, y, iterations=1000, learning_rate=0.5, display=False, reinitialize=False, regChange=0.01, regularize=False):
        # neural net training

        self.gradient_adjust(X, y, iterations=iterations, learning_rate=learning_rate, regChange=regChange,
                             display=display, regularize=regularize)

        if self.accuracy(X, y) < 0.70 and reinitialize:
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

    def squared_error(self, X, y):
        # produces the total squared error of neural net
        error = 0.0
        yHat = self.predict(X)
        for i in range(len(yHat)):
            error += ((y[i] - yHat[i]) ** 2) * 0.5

        return error

