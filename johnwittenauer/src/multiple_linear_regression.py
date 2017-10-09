# coding=utf-8
"""
http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-2/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.lib import computeCost, gradient_descent


if __name__ == '__main__':
    path = os.getcwd() + "\data\ex1data2.txt"
    data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print (data2.head())
    # add ones column
    data2.insert(0, 'Ones', 1)
    # set X (training data) and y (target variable)
    cols = data2.shape[1]
    X2 = data2.iloc[:,0:cols-1]
    y2 = data2.iloc[:,cols-1:cols]

    # convert to matrices and initialize theta
    X2 = np.matrix(X2.values)
    y2 = np.matrix(y2.values)
    # theta2 = np.matrix(np.array([0,0,0]))

    theta2 = np.matrix(np.zeros(X2.shape[1]))
    theta2 = theta2.T
    print(X2.shape, theta2.shape, y2.shape)
    # perform linear regression on the data set
    alpha = 0.001
    iters = 30
    g2, cost2, final_cost2 = gradient_descent(X2, y2, theta2, alpha, iters)

    # get the cost (error) of the model
    print (computeCost(X2, y2, g2))