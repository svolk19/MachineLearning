import numpy as np
import matplotlib.pyplot as plt
import utils.activations as act
import utils.gradient_descents as gd


class neural_network(object):
    def __init__(self, input_size, output_size, hidden1_size, hidden2_size):

        # layer sizes
        self.inputSize = input_size
        self.outputSize = output_size
        self.hidden1Size = hidden1_size
        self.hidden2Size = hidden2_size

        # initialize random primary weight and bias scheme
        w1 = np.random.randn(self.inputSize, self.hidden1Size)
        w2 = np.random.randn(self.hidden1Size, self.hidden2Size)
        w3 = np.random.randn(self.hidden2Size, self.outputSize)

        self.weights = np.array([w1, w2, w3])

        b1 = np.zeros(self.hidden1Size)
        b2 = np.zeros(self.hidden2Size)
        b3 = np.zeros(self.outputSize)

        self.biases = np.array([b1, b2, b3])

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

        return act.softmax(self.z4, predict=True)

    def forwardPropagate(self, X):

        # feed forward data set X
        # assumes X is a numpy array, returns numpy array
        self.z2 = np.dot(X, self.weights[0]) + self.biases[0]
        self.a2 = act.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights[1]) + self.biases[1]
        self.a3 = act.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.weights[2]) + self.biases[2]

        return act.softmax(self.z4)

    def backPropagate(self, X, y):

        # backpropagation of value X
        num_examples = len(X)
        yhat = self.forwardPropagate(X)
        delta4 = -1.0 * (y - yhat)
        dJdW3 = np.dot(self.a3.T, delta4)
        dJdB3 = np.sum(delta4, axis=0)
        delta3 = np.dot(delta4, self.weights[2].T) * act.sigmoid(self.z3, deriv=True)
        dJdW2 = np.dot(self.a2.T, delta3)
        dJdB2 = np.sum(delta3, axis=0)
        delta2 = np.dot(delta3, self.weights[1].T) * act.sigmoid(self.z2, deriv=True)
        dJdW1 = np.dot(X.T, delta2)
        dJdB1 = np.sum(delta2, axis=0)

        weight_gradients = np.array([dJdW1, dJdW2, dJdW3])
        bias_gradients = np.array([dJdB1, dJdB2, dJdB3])

        return weight_gradients, bias_gradients

    def train(self, X, y, iterations=1000, learning_rate=0.5, reg_lambda=0.01, regularize=False):
        # neural net training

        weight_gradients, bias_gradients = self.backPropagate(X, y)

        self.weights, self.biases = gd.batch_gradient_descent(self.weights, self.biases, weight_gradients, bias_gradients,
                                                              X, y, iterations=iterations, learning_rate=learning_rate,
                                                              reg_lambda=reg_lambda, regularize=regularize)

    def accuracy(self, X, y, string=False):
        # produces the accuracy of neural net
        errorCount = 0
        yhat = self.predict(X)
        for i, elem in enumerate(yhat):
            if y[i][elem] != 1:
                errorCount += 1

        accuracy = 1.0 - errorCount / len(y)

        if string:
            print("accuracy: " + str(accuracy * 100.0) + "%")
        else:
            return accuracy

