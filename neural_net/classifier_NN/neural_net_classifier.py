import numpy as np
import matplotlib.pyplot as plt
import neural_net.utils.activations as act
<<<<<<< HEAD

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

        self.weights = np.array([self.w1, self.w2, self.w3])

        self.b1 = np.zeros(self.hidden1Size)
        self.b2 = np.zeros(self.hidden2Size)
        self.b3 = np.zeros(self.outputSize)
=======


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
        # feed forward dataset X
        # assumes X is a numpy array, returns numpy array
        self.z2 = np.dot(X, self.weights[0]) + self.biases[0]
        self.a2 = act.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights[1]) + self.biases[1]
        self.a3 = act.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.weights[2]) + self.biases[2]
<<<<<<< HEAD
        results = np.exp(self.z4)
        probabilities = results / np.sum(results, axis=1, keepdims=True)
        return np.argmax(probabilities, axis=1)
=======

        return act.softmax(self.z4, predict=True)
>>>>>>> master

    def forwardPropagate(self, X):

        # feed forward data set X
        # assumes X is a numpy array, returns numpy array
<<<<<<< HEAD
        self.z2 = np.dot(X, self.w1) + self.b1
        self.a2 = act.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2) + self.b2
        self.a3 = act.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.w3) + self.b3
        results = np.exp(self.z4)
        probabilities = results / np.sum(results, axis=1, keepdims=True)
        return probabilities
=======
        self.z2 = np.dot(X, self.weights[0]) + self.biases[0]
        self.a2 = act.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights[1]) + self.biases[1]
        self.a3 = act.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.weights[2]) + self.biases[2]

        return act.softmax(self.z4)
>>>>>>> master

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

<<<<<<< HEAD
    def gradient_adjust(self, X, y, iterations=1000, learning_rate=0.5, regChange=0.01, display=False,
                            regularize=False):

        # train neural network
        for num in range(iterations):

            # calculate gradients
            dJdw1, dJdw2, dJdw3, dJdB1, dJdB2, dJdB3 = self.backPropagate(X, y)
=======
        return weight_gradients, bias_gradients

    def batch_gradient_descent(self, weight_gradients, bias_gradients, X, y, iterations=5000,
                                   learning_rate=0.01, reg_lambda=0.01, regularize=False, print_accuracies=False):

        """
        implements simple batch gradient descent
        :param weights: tuple of numpy weight arrays
        :param biases: tuple of numpy bias arrays
        :param weight_gradients: tuple of numpy arrays of weight gradients
        :param bias_gradients: tuple of numpy arrays of bias gradients
        :param X: input data
        :param y: output data
        :param print_accuracies: prints accuracy at each iteration
        :param reg_lambda: regularization rate
        :return: new weights, new biases
        """
>>>>>>> master

        for num in range(iterations):

            # iterate through weights and add regularization
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

            if display:
                plt.scatter(num, self.accuracy(X, y))

        if display:
            plt.show()

    def train(self, X, y, iterations=1000, learning_rate=0.5, display=False, reinitialize=False, regChange=0.01,
              regularize=False):
        # neural net training

        self.gradient_adjust(X, y, iterations=iterations, learning_rate=learning_rate, regChange=regChange,
                             display=display, regularize=regularize)
=======
                for weight_scheme in self.weights:
                    for weight in weight_scheme:
                        weight += reg_lambda * weight

            # iterate through weight gradients and adjust weights
            for i, weight_gradient_scheme in enumerate(weight_gradients):
                for j, weight_gradient in enumerate(weight_gradient_scheme):
                    self.weights[i][j] -= learning_rate * weight_gradient

            # iterate through bias gradients and adjust biases
            for i, bias_gradient_scheme in enumerate(bias_gradients):
                for j, bias_gradient in enumerate(bias_gradient_scheme):
                    self.biases[i][j] -= learning_rate * bias_gradient

            if print_accuracies:
                print(self.accuracy(X, y))

    def train(self, X, y, iterations=1000, learning_rate=0.5, reg_lambda=0.01, regularize=False, print_accuracies=False):
        # neural net training

        weight_gradients, bias_gradients = self.back_propagate(X, y)

        self.batch_gradient_descent(weight_gradients, bias_gradients, X, y, iterations=iterations,
                                    learning_rate=learning_rate, reg_lambda=reg_lambda, regularize=regularize,
                                    print_accuracies=print_accuracies)
>>>>>>> master

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

    def squared_error(self, X, y):
        # produces the total squared error of neural net
        error = 0.0
        yHat = self.predict(X)
        for i in range(len(yHat)):
            error += ((y[i] - yHat[i]) ** 2) * 0.5

        return error

