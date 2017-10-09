# coding=utf-8
"""
http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-2/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


def normal_equations(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


if __name__ == '__main__':
    path = os.getcwd() + '\data\ex1data2.txt'
    data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print(data2.head())
    print(data2.describe())

    data2 = (data2 - data2.mean()) / data2.std()  # Size   Bedrooms Price 各自用mean std来计算
    # data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    # data2 = (data2 - data2.mean()) / (data2.max() - data2.min())

    print(data2.head())
    print(data2.describe())

    # add ones column
    data2.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data2.shape[1]
    X2 = data2.iloc[:, 0:cols - 1]
    y2 = data2.iloc[:, cols - 1:cols]

    # convert to matrices and initialize theta
    X2 = np.matrix(X2.values)
    y2 = np.matrix(y2.values)
    theta2 = np.matrix(np.array([0, 0, 0]))

    print("shapes:", X2.shape, y2.shape, theta2.shape)

    # normal_equation
    normal_equation_theta = normal_equations(X2, y2)
    print("normal_equation_theta:", normal_equation_theta)
    print("normal equation cost:", computeCost(X2, y2, normal_equation_theta.T))

    # initialize variables for learning rate and iterations
    alpha = 0.01
    iters = 10000

    # perform linear regression on the data set
    g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

    print(g2)
    print("g2.shape:", g2.shape)
    print("cost:", cost2[-1])

    # get the cost (error) of the model
    print("cost:", computeCost(X2, y2, g2))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')

    plt.show()
