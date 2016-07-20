import numpy as np
import matplotlib.pyplot as plt

class neural_network(object):
    def __init__(self, inputsize, outputsize, hidden1size, hidden2size):

        # layer sizes
        self.inputSize = inputsize
        self.outputSize = outputsize
        self.hidden1Size = hidden1size
        self.hidden2Size = hidden2size

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

    def softmax(self, X):

        # activation for last set of neurons: the probabilistic normalizer
        # assumes X is a numpy array
        exp = np.exp(X)
        expSum = np.sum(exp, axis=0)
        return exp / expSum

    def predict(self, X):

        # feed forward data set X
        # assumes X is a numpy array, returns numpy array
        self.z2 = np.dot(X, self.w1) + self.b1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2) + self.b2
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.w3) + self.b3
        results = np.exp(self.z4)
        probabilities = results / np.sum(results, axis=1, keepdims=True)
        return np.argmax(probabilities, axis=1)

    def forwardPropagate(self, X):

        # feed forward data set X
        # assumes X is a numpy array, returns numpy array
        self.z2 = np.dot(X, self.w1) + self.b1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2) + self.b2
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.w3) + self.b3
        results = np.exp(self.z4)
        probabilities = results / np.sum(results, axis=1, keepdims=True)
        return probabilities

    def backPropagate(self, X, y):

        # backpropagation of value X
        num_examples = len(X)
        yhat = self.forwardPropagate(X)
        delta4 = -1.0 * (y - yhat)
        dJdW3 = np.dot(self.a3.T, delta4)
        dJdB3 = np.sum(delta4, axis=0)
        delta3 = np.dot(delta4, self.w3.T) * self.sigmoid(self.z3, deriv=True)
        dJdW2 = np.dot(self.a2.T, delta3)
        dJdB2 = np.sum(delta3, axis=0)
        delta2 = np.dot(delta3, self.w2.T) * self.sigmoid(self.z2, deriv=True)
        dJdW1 = np.dot(X.T, delta2)
        dJdB1 = np.sum(delta2, axis=0)

        return dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3

    def gradient_adjust(self, X, y, iterations=5000, learning_rate=0.01, regChange=0.01, display=False, regularize=False):

        # train neural network until greater than or equal to 99.5% accuracy is achieved
        for num in range(iterations):

            # calculate gradients
            dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3 = self.backPropagate(X, y)

            # train weights and biases

            if regularize:
                self.w1 += regChange * self.w1
                self.w2 += regChange * self.w2
                self.w3 += regChange * self.w3

            for i, deriv in enumerate(dJdW1):
                self.w1[i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdW2):
                self.w2[i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdW3):
                self.w3[i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdB1):
                self.b1[i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdB2):
                self.b2[i] += -learning_rate * deriv

            for i, deriv in enumerate(dJdB3):
                self.b3[i] += -learning_rate * deriv


            if display:
                plt.scatter(num, self.accuracy(X, y))

        if display:
            plt.show()

    def train(self, X, y, iterations=1000, learning_rate=0.5, regChange=0.01, display=False,
              regularize=False, reinitialize=False):
        # neural net training

        self.gradient_adjust(X, y, iterations=iterations, learning_rate=learning_rate, regChange=regChange,
                             display=display, regularize=regularize)

        if reinitialize and self.accuracy(X, y) < 0.95:
            # try a different random weighting
            self.w1 = np.random.randn(self.inputSize, self.hidden1Size)
            self.w2 = np.random.randn(self.hidden1Size, self.hidden2Size)
            self.w3 = np.random.randn(self.hidden2Size, self.outputSize)

            self.train(X, y)


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

