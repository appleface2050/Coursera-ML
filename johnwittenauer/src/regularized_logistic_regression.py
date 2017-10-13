# coding=utf-8

"""
http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import scipy.optimize as opt
import os

from logistic_regression import sigmoid, predict


def feature_mapping(x, y, power, as_ndarray=False):
    """return mapped features as ndarray or dataframe"""
    # data = {}
    # # inclusive
    # for i in np.arange(power + 1):
    #     for p in np.arange(i + 1):
    #         data["f{}{}".format(i - p, p)] = np.power(x, i - p) * np.power(y, p)

    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
            for i in np.arange(power + 1)
            for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)



def gradient(theta, X, y):
    '''just 1 batch gradient'''
    if len(y.shape) == 2:
        y = np.array([i[0] for i in np.array(y2).tolist()])
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


def regularized_gradient(theta, X, y, l=1):
    '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term


def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def regularized_cost(theta, X, y, l=1):
    '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])

    return grad


def costReg(theta, X, y, learningRate):  # learningRate = λlambda
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg


if __name__ == '__main__':

    path = "D:\git\Coursera-ML\johnwittenauer\data\ex2data2.txt"
    data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

    positive = data2[data2['Accepted'].isin([1])]
    negative = data2[data2['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    # plt.show()

    degree = 5
    x1 = data2['Test 1']
    x2 = data2['Test 2']

    data2.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(0, i):
            # data2['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
            data2['F' + str(i - j) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

    # data2 = data2.drop('Test 1', axis=1)
    data2.drop('Test 1', axis=1, inplace=True)
    data2.drop('Test 2', axis=1, inplace=True)

    print(data2.head())
    # set X and y (remember from above that we moved the label to column 0)
    cols = data2.shape[1]
    X2 = data2.iloc[:, 1:cols]
    y2 = data2.iloc[:, 0:1]

    X2 = feature_mapping(x1, x2, power=6, as_ndarray=True)


    # convert to numpy arrays and initalize the parameter array theta
    # X2 = np.array(X2.values)
    y2 = np.array(y2.values)
    theta2 = np.zeros(X2.shape[1])

    learningRate = 1

    # print(costReg(theta2, X2, y2, learningRate))
    result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
    final_theta = result2[0]
    print("final theta:", final_theta)
    print("cost Reg:", costReg(result2[0], X2, y2, learningRate))
    # report
    theta_min = np.matrix(final_theta)
    predictions = predict(theta_min, X2)
    # print(predictions)
    print("report2:", classification_report(y2, predictions))

    # #####################
    # theta = np.zeros(X2.shape[1])
    # print(X2.shape, y2.shape, theta.shape)
    # res = opt.minimize(fun=regularized_cost, x0=theta, args=(X2, y2), method='Newton-CG', jac=regularized_gradient)
    # final_theta = res.x
    # predictions2 = predict(final_theta, X2)
    # print("report2:",classification_report(y2, predictions2))