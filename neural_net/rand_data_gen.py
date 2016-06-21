def createScaledGravityDataset():
    import numpy as np
    import random
    g = 9.80665

    X = []
    while len(X) < 50:
        newInts = (random.randint(0, 100), random.randint(0, 100))
        if -0.5 * g * (newInts[0] ** 2) + newInts[1] * newInts[0] > 0:
            X.append(newInts)

    X = np.array(X)
    y = np.empty(50, dtype=float)

    for t, vNot in X:
        elem = -0.5 * g * (t ** 2) + vNot * t
        np.append(y, elem)

    X = X/np.amax(X, axis=0)
    y = y/np.amax(y, axis = 0)

    return X, y