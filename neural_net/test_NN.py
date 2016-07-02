# creates a valid (initial height, initial velocity) coordinate pair with a label time value such
# that time occurs when height = 0. Calculates data based on simple newtonian equation:
#  h = -1/2 g t^2 + Vo * t + Ho

import numpy as np

def createGravData():

    g = 9.80665

    #initialize an empty numpy array
    X_dat = np.empty([100, 2])

    #fill array with random x values
    for i in range(100):
        X_pair = np.random.randint(20, 100, 2)
        X_dat[i] = X_pair

    #calculate y value (time) for each random X, and add it to a new array with corresponding indicies
    y_list = []

    for i, X in enumerate(X_dat):
        y = (X[1] + np.sqrt([(X[1] ** 2) + (2 * g * X[0])])) / g
        y_list.append(y)

    y_dat = np.array(y_list)

    return X_dat, y_dat


def NN_accuracy(NN):
    # create a dataset
    X, y = createGravData()

    yHat = NN.forward(X)

    #find avg squared error
    errorArray = np.subtract(y, yHat)

    return np.sum(np.square(errorArray)) / np.sum(np.square(y))

print(createGravData())
