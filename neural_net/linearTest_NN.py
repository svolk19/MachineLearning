import numpy as np
import argparse

#randomly generate linnear data with numpy equation: y = mx + b

def linearData(size=100):

    randM = np.random.randint(0, 100)
    randB = np.random.randint(0, 100)

    #initialize an empty numpy array
    X_dat = np.empty([size, 2])

    #fill array with random x values
    for i in range(size):
        X_pair = np.random.randint(0, 100, 2)
        X_dat[i] = X_pair

    #calculate y value for each random X, and add it to a new array with corresponding indicies
    y_list = []
    y_dat = np.zeros(size)
    for X in X_dat:
        y = (X[1] * randM) + randB
        y_list.append(y)

    for i in enumerate(y_list):
        y_dat[i]

    return X_dat, y_dat

def NN_accuracy(NN):
    # create a dataset
    X, y = linearData()

    yHat = NN.forward(X)

    #find avg squared error
    errorArray = np.subtract(y, yHat)

    return np.sum(np.square(errorArray)) / np.sum(np.square(y))

print(linearData())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("size", help="number of random linear values", type=int)
    size = parser.parse_args()
    linearData(size)

