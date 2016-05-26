import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
import pandas as pd
from collections import Counter

style.use('fivethirtyeight')

def distance(point1, point2):
    """
    :param point1: a tuple or list containing a cartesian coordinate set of infinite dimension
    :param point2: a tuple or list containing a cartesian coordinate set of infinite dimension
    :return: distance (a float) between the two points
    """

    #find the total of the terms inside the distance formula:
    # d = sqrt(term1 + term2 + term3 ... + termN)
    # where either term is the difference of the dimension terms of the cartesian pairs squared
    termTotal = 0
    for index in range(len(point1)):
        term = (point2[index] - point1[index]) ** 2
        termTotal += term

    #sqrt termTotal to finish euclidian distance formula
    return math.sqrt(termTotal)

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

def KNearestNeighbors(data, point, K):
    """
    :param data: a series of cartesian points with label
    :param point: a cartesian point with the same number of dimensions as an object in data
    :param K: the number of 'neighbor' points to compare 'point' to to predict point's class
    :return: the predicted class of point (an int--2 or 4)
    """
    count = 0
    closePointList = []
    while count < K:
        closestDistance, closePoint = 0
        for dataPoint in data:
            if closestDistance == 0:
                closePoint, closestDistance = dataPoint, distance(dataPoint, point)
            else:
                newDistance = distance(dataPoint, point)
                if newDistance < closestDistance:
                    closestDistance, closePoint = newDistance, dataPoint
        closePointList.append(closePoint)
        count += 1














        













