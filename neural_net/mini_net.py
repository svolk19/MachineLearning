import numpy as np
import neural_net.utils.activations as act

X = np.array(([0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15], [6, 16], [7, 17], [8, 18], [9, 19]), dtype=float)
y = np.array(([10], [15], [20], [25], [30], [35], [40], [45], [50], [55]), dtype=float)

# Normalize
X = X / np.amax(X, axis=0)
y = y / np.amax(y, axis=0)


class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        w1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        w2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

        self.weights = np.array([w1, w2])

        b1 = np.zeros(self.hiddenLayerSize)
        b2 = np.zeros(self.outputLayerSize)

        self.biases = np.array([b1, b2])

        self.z2 = 0.0
        self.a2 = 0.0
        self.z3 = 0.0

    def forward_propagate(self, X):
        self.z2 = np.dot(X, self.weights[0]) + self.biases[0]
        self.a2 = act.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights[1]) + self.biases[1]
        yhat = act.sigmoid(self.z3)
        return yhat

    def back_propagate(self, X, y):
        yhat = self.forward_propagate(X)

        delta3 = np.multiply(-(y - yhat), act.sigmoid(self.z3, deriv=True))
        dJdW2 = np.dot(self.a2.T, delta3)
        dJdB2 = np.sum(delta3, axis=0)

        delta2 = np.dot(delta3, self.weights[1].T) * act.sigmoid(self.z2, deriv=True)
        dJdW1 = np.dot(X.T, delta2)
        dJdB1 = np.sum(delta2, axis=0)

        weight_gradients = np.array([dJdW1, dJdW2])
        bias_gradients = np.array([dJdB1, dJdB2])

        return weight_gradients, bias_gradients

    def gradient_adjust(self, X, y, iterations=1000, learning_rate=0.5):

        # train neural network until greater than or equal to 99.5% accuracy is achieved
        for num in range(iterations):

                # calculate gradients
                weight_gradients, bias_gradients = self.back_propagate(X, y)

                # train weights and biases
                for i, weight_gradient_scheme in enumerate(weight_gradients):
                    for j, weight_gradient in enumerate(weight_gradient_scheme):
                        self.weights[i][j] -= learning_rate * weight_gradient

                for i, bias_gradient_scheme in enumerate(bias_gradients):
                    for j, bias_gradient in enumerate(bias_gradient_scheme):
                        self.biases[i][j] -= learning_rate * bias_gradient


    def train(self, X, y, iterations=1000, learning_rate=0.01):
        # neural net training

        self.gradient_adjust(X, y, iterations=iterations, learning_rate=learning_rate)

    def accuracy(self, X, y, string=False):
        # produces the accuracy of neural net

        yhat = self.forward_propagate(X)

        error_sum = np.sum(np.absolute(np.subtract(yhat, y)))
        y_sum = np.sum(np.absolute(y))
        accuracy = 1.0 - error_sum / y_sum

        if string:
            print("accuracy: " + str(accuracy * 100.0) + "%")
        else:
            return accuracy


NN = Neural_Network()
NN.train(X, y, iterations=100000, learning_rate=0.01)

print(NN.accuracy(X, y, string=True), '\n', '\nweight 1:', NN.weights[0], '\n\nweight2: ', NN.weights[1],
      '\n\nbias 1:', NN.biases[0], '\n\nbias 2:', NN.biases[1])



