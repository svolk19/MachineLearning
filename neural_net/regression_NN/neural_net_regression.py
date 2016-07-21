import numpy as np
import matplotlib.pyplot as plt
import neural_net.utils.activations as act
<<<<<<< HEAD
=======

class NeuralNetwork(object):
>>>>>>> master

    def __init__(self, inputSize, outputSize, hidden1Size, hidden2Size):

        # layer sizes
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hidden1Size = hidden1Size
        self.hidden2Size = hidden2Size

        # initialize random primary weight and bias scheme
<<<<<<< HEAD
        self.w1 = np.random.randn(self.inputSize, self.hidden1Size)
        self.w2 = np.random.randn(self.hidden1Size, self.hidden2Size)
        self.w3 = np.random.randn(self.hidden2Size, self.outputSize)

        self.weights = np.array([self.w1, self.w2, self.w3])

        self.b1 = np.zeros(self.hidden1Size)
        self.b2 = np.zeros(self.hidden2Size)
        self.b3 = np.zeros(self.outputSize)
=======
        w1 = np.random.randn(self.inputSize, self.hidden1Size)
        w2 = np.random.randn(self.hidden1Size, self.hidden2Size)
        w3 = np.random.randn(self.hidden2Size, self.outputSize)

        self.weights = np.array([w1, w2, w3])

        b1 = np.zeros(self.hidden1Size)
        b2 = np.zeros(self.hidden2Size)
        b3 = np.zeros(self.outputSize)

        self.biases = np.array([b1, b2, b3])
>>>>>>> master

        self.biases = np.array([self.b1, self.b2, self.b3])

        # initialize forward propagation parameters to 0.0
        self.z2 = 0.0
        self.a2 = 0.0
        self.z3 = 0.0
        self.a3 = 0.0
        self.z4 = 0.0
        self.a4 = 0.0

    def predict(self, X):
        # feed forward data set X
        # assumes X is a numpy array, returns numpy array
        self.z2 = np.dot(X, self.weights[0]) + self.biases[0]
        self.a2 = act.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights[1]) + self.biases[1]
        self.a3 = act.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.weights[2]) + self.biases[2]
        self.a4 = act.sigmoid(self.z4)
        return self.a4

    def back_propagate(self, X, y):

        # back propagation of value
        yHat = self.predict(X)

<<<<<<< HEAD
        delta4 = np.multiply(-(y - self.yHat),
=======
        delta4 = np.multiply(-(y - yHat),
>>>>>>> master
                             act.sigmoid(self.z4, deriv=True))
        dJdW3 = np.dot(self.a3.T, delta4)
        dJdB3 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = np.dot(delta4, self.weights[2].T) * act.sigmoid(self.z3, deriv=True)
        dJdW2 = np.dot(self.a2.T, delta3)
        dJdB2 = np.sum(delta3, axis=0)
        delta2 = np.dot(delta3, self.weights[1].T) * act.sigmoid(self.z2, deriv=True)
        dJdW1 = np.dot(X.T, delta2)
        dJdB1 = np.sum(delta2, axis=0)

        return dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3

    def gradient_adjust(self, X, y, iterations=1000, learning_rate=0.01, reg_lambda=0.01, display=False,
                        regularize=False):

        # train neural network
        for num in range(iterations):

            # calculate gradients
            dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3 = self.back_propagate(X, y)

            # train weights and biases

            if regularize:
<<<<<<< HEAD
                self.weights[0] += regChange * self.weights[0]
                self.weights[1] += regChange * self.weights[1]
                self.weights[2] += regChange * self.weights[2]

            for i, deriv in enumerate(dJdw1):
                self.weights[0][i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdw2):
                self.weights[1][i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdw3):
                self.weights[2][i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdB1):
                self.biases[0][i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdB2):
                self.biases[1][i] -= learning_rate * deriv

            for i, deriv in enumerate(dJdB3):
                self.biases[2][i] -= learning_rate * deriv
=======
                self.weights[0] += reg_lambda * self.weights[0]
                self.weights[1] += reg_lambda * self.weights[1]
                self.weights[2] += reg_lambda * self.weights[2]

            for i, deriv in enumerate(dJdW1):
                self.weights[0][i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdW2):
                self.weights[1][i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdW3):
                self.weights[2][i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdB1):
                self.biases[0][i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdB2):
                self.biases[1][i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdB3):
                self.biases[2][i] += -learning_rate * deriv
>>>>>>> master

            if display:
                plt.scatter(num, self.accuracy(X, y))

        if display:
            plt.show()

<<<<<<< HEAD
    def train(self, X, y, iterations=1000, learning_rate=0.5, display=False, reinitialize=False, regChange=0.01, regularize=False):
        # neural net training

        self.gradient_adjust(X, y, iterations=iterations, learning_rate=learning_rate, regChange=regChange,
                             display=display, regularize=regularize)

        return 'accuracy:' + str(self.accuracy(X, y))
=======
    def train(self, X, y, iterations=1000, learning_rate=0.5, reg_lambda=0.01, regularize=False, print_accuracies=False, display=False):
            # neural net training

            # self.batch_gradient_descent(X, y, iterations=iterations,
            #                             learning_rate=learning_rate, reg_lambda=reg_lambda, regularize=regularize,
            #                             print_accuracies=print_accuracies)
>>>>>>> master

        self.gradient_adjust(self, X, y, iterations=iterations, learning_rate=learning_rate, reg_lambda=reg_lambda, display=display,
                    regularize=regularize)

    def accuracy(self, X, y, string=False):
        # produces the accuracy of neural net
        yhat = self.predict(X)
        error_difference = np.absolute(np.subtract(yhat, y))
        error = np.absolute(np.divide(error_difference, y))
        accuracy = 1.0 - (np.sum(error) / len(error))

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