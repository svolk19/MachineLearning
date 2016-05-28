import matplotlib.pyplot as plt
import random

class data():
    """
    Represents data set of (x,y) pairs
    """
    def __init__(self, dataList):
        """
        Initializes lists, an x list, and a y value list with corresponding indicies
        Initializes a dictionary of (x, y) pairs as self.dataDict :param dataList
        """
        xList = []
        yList = []

        for index in range(0, len(dataList)):
            xList.append(dataList[index][0])
            yList.append(dataList[index][1])
        self.xList = xList
        self.yList = yList
        self.dataList = dataList

    def getXAvg(self):
        """
        :return: average value of x
        """
        total = 0.0
        for elem in self.xList:
            total += elem
        avg = float(total) / float(len(self.xList))
        return avg

    def getYAvg(self):
        """
        :return: average value of y
        """
        total = 0.0
        for elem in self.yList:
            total += elem
        avg = float(total) / float(len(self.yList))
        return avg

    def xyMultiAvg(self):
        """
        :return: average value of the multiplication of every corresponding x, y pair
        (to be used in linearRegression function)
        """
        multiList = []
        for index in range(0, len(self.xList)):
            multiList.append(float(self.xList[index]) * float(self.yList[index]))


        total = 0.0
        for elem in multiList:
            total += elem

        return total / float(len(self.yList))

    def getXList(self):
        return self.xList

    def getYList(self):
        return self.yList

def getAvg(ListObj):
    """
    :return: average value of y
    """
    total = 0.0
    for elem in ListObj:
        total += elem
    avg = float(total) / float(len(ListObj))
    return avg

def LinearRegression(data):
    """
    :param data: object of type data
    :return: the best fit line (y = mx + b) to the data set
    """
    m = float((data.getXAvg() * data.getYAvg()) - data.xyMultiAvg()) / ((data.getXAvg() ** 2) - getAvg([i ** 2 for i in data.xList]))
    b = data.getYAvg() - m * data.getXAvg()
    return m, b

def line(m, b, xVal):
    yVal = m * xVal + b
    return yVal

def rSqr(regressionLine, data, m, b):

    #calculate total squared error for regression line
    totalRegDiff = 0.0
    for i in range(0, len(data.yList)):
        totalRegDiff += ((regressionLine(m, b, data.getXList()[i]) - data.getYList()[i]) ** 2)

    #calculate total squared error for average of y line
    totalAvgDiff = 0.0
    yAvg = float(data.getYAvg())
    for i in range(0, len(data.yList)):
        totalAvgDiff += ((regressionLine(m, b, data.getXList()[i]) - yAvg) ** 2)

    #return the coefficient of determination (r squared)
    return 1.0 - totalRegDiff / totalAvgDiff

def create_dataset(number, variance, step = 1, correlation = 'False'):
    """
    :param number: number of data points
    :param variance: maximum positive or negative difference
    :param step: used for correlation, the approximate 'slope' of correlation
    :param correlation: none, 'pos', or 'neg', correlation of data set
    :return: object of type data randomly generated that meets specified parameters
    """
    val = 1
    dataList = []
    for i in range(number):
        y = val + random.randrange(-variance, variance)
        dataList.append((i, y))
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    return data(dataList)

dataSet = create_dataset(100, 60, correlation = 'pos')
plt.scatter(dataSet.getXList(), dataSet.getYList())

m, b = LinearRegression(dataSet)
plt.plot([dataSet.getXList()[0], dataSet.getXList()[-1]], [line(m, b, dataSet.getXList()[0]), line(m, b, dataSet.getXList()[-1])])
plt.show()

print(rSqr(line, dataSet, m, b))