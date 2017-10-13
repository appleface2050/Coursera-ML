# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg


if __name__ == '__main__':
    data = loadmat('D:\git\Coursera-ML\johnwittenauer\data\ex3data1.mat')
    print(data)
    print(type(data))
    print(data['X'].shape, data['y'].shape)
