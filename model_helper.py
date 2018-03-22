from __future__ import unicode_literals
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_test_case():
    with open('test_case.pickle', 'rb') as fp:
        return pickle.load(fp)


def divide_data(X,Y):
    return train_test_split(X, Y,test_size = 0.05, random_state = 42)


def model_diff(predict,test_y):
    temp = (predict - test_y)
    return np.nan_to_num(temp)


if __name__ == '__main__':
    out = load_test_case()
    X = out['X']
    Y = out['Y']
    pos = out['Pos']
    X_train, X_test, y_train, y_test= divide_data(X,Y)

    y_predict = y_test*1.02
    plt.hist(model_diff(y_predict,y_test))
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    fig = plt.gcf()
    fig.show()


    plt.plot(y_predict,y_test)
    fig.show()
