import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
import pandas as pd
from collections import Counter
import data

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
df.drop(['class'], 1, inplace = True)

cartesianList = []
proxyList = []
for index in range(len(df['mitoses'])):
    for elem in df:
        proxyList.append(int(df[elem][index]))
    cartesianList.append(proxyList)
    proxyList = []

def KNearestNeighbors(data, point, K):
    """
    :param data: a series of cartesian points with label
    :param point: a cartesian point with the same number of dimensions as an object in data
    :param K: the number of 'neighbor' points to compare 'point' to to predict point's class
    :return: the predicted class of point (an int--2 or 4)
    """

    #find the K closest points in data to point
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

    #sort the list of close points by class, and return the class most prevalent in closePointList
    classCount1, classCount2 = []
    for index in range(len(closePointList)):
        if index == 0:
            classCount1.append(getClass(closePointList[0]))
        elif getClass(closePointList[index]) == classCount1[0]:
            classCount1.append(getClass(closePointList[index]))
        else:
            classCount2.append(getClass(closePointList[index]))

    #return most prevalent class
    if len(classCount1) > len(classCount2):
        return classCount1[0]
    else:
        return classCount2[0]