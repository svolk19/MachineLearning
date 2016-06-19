import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

#Generate dataset
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise = 0.20)
plt.scatter(X[:, 0], X[:, 1], s = 40, c = y, cmap = plt.cm.spectral)

class neural_net():
    def __init__(self, hl_dim):
        #initializes neural net with single hidden layer with hl_dim (an int) neurons
        self.inputLayer = 
