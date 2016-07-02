import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class neural_network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize1 = 5
        self.hiddenLayerSize2 = 5
        self.w1 = np.random.randn(self.inputLayerSize,
                                  self.hiddenLayerSize1)
        self.w2 = np.random.randn(self.hiddenLayerSize1,
                                  self.hiddenLayerSize2)
        self.w3 = np.random.randn(self.hiddenLayerSize2,
                                  self.outputLayerSize)

    def forward(self, x):
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.w3)
        yHat = self.sigmoid(self.z4)
        return yHat

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)


    def costFunction(self, X, y):
        yHat = self.forward(X)
        return 0.5 * sum((y - yHat) ** 2)

    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)

        delta4 = np.multiply(-(y - self.yHat),
                             self.sigmoidPrime(self.z4))

        dJdW3 = np.dot(self.a3.T, delta4)

        delta3 = np.dot(delta4, self.w3.T) * self.sigmoidPrime(self.z3)

        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.w2.T) * self.sigmoidPrime(self.z2)

        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2, dJdW3

    def sigmoidPrime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z) ** 2))

    def tanhPrime(self, z):
        return (2 / (np.exp(z) + np.exp(-z))) ** 2

    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.w1.ravel(), self.w2.ravel(), self.w3.ravel()))
        return params

    def setParams(self, params):
        # Set W1, W2, and W3 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize2 * self.inputLayerSize
        self.w1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize1))
        W2_end = W1_end + self.hiddenLayerSize1 * self.hiddenLayerSize2
        self.w2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize1, self.hiddenLayerSize2))
        W3_end = W2_end + self.hiddenLayerSize2 * self.outputLayerSize
        self.w3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayerSize2, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)

        return cost, grad

    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : False}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

    def show(self):
        plt.plot(T.J)
        plt.grid(1)
        plt.show()

import test_NN as tNN

X, y = tNN.createGravData()
X = X/np.amax(X, axis=0)
y = y/np.amax(y, axis=0)

NN = neural_network()
T = trainer(NN)
T.train(X, y)
accuracy = tNN.NN_accuracy(NN)
print('accuracy: %f' %accuracy)

