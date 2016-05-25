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


newDataList = []
for item in df:
    newDataList.append(item)

print(newDataList['clump_thickness'])