import neural_net.utils.csv_reader as csv
import numpy as np
import neural_net.classifier_NN.neural_net_classifier as neural_net
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


def get_classes(labels):
    # returns list of data classes in labels

    label_list = []
    for i in labels:
        if i not in label_list:
            label_list.append(i)

    return label_list


def get_class_index(class_list, class_item):
    # return index of class_item in class_list

    for i, item in enumerate(class_list):
        if item == class_item:
            return i


def NN_train(labels, data):
    # test data classification problem

    X = data
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

    class_list = get_classes(y)

    y_trainList = []
    for elem in y_train:
        class_label_list = [0 for i in range(len(class_list))]
        class_label_list[get_class_index(class_list, elem)] = 1
        y_trainList.append(class_label_list)

    y_train = np.array(y_trainList)
    y_train.reshape((len(y_trainList), len(class_list)))

    y_testList = []
    for elem in y_test:
        class_label_list = [0 for i in range(len(class_list))]
        class_label_list[get_class_index(class_list, elem)] = 1
        y_testList.append(class_label_list)

    y_test = np.array(y_testList)
    y_test.reshape((len(y_testList), len(class_list)))

    NN = neural_net.NeuralNetwork(len(data[0]), len(get_classes(labels)), 13, 13)
    NN.train(X_train, y_train, learning_rate=0.01, iterations=1000, display=True)
    print(NN.accuracy(X_test, y_test))


def main():
    csv_filepath = "C:/Users/sam/Desktop/breast_cancer_wisconsin.csv"
    index_location = 10

    labels, data = csv.csv_reader(csv_filepath, index_location)

    NN_train(labels, data)

if __name__ == "__main__":
    main()


