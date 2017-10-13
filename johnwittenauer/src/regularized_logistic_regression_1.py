# coding=utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.append('..')

from sklearn.metrics import classification_report
import scipy.optimize as opt

def gradient(theta, X, y):
    '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)


def get_X(df):
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)  # column concat
    return data.iloc[:, :-1].as_matrix()  # this return ndarray, not matrix


def get_y(df):
    '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])


def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def regularized_cost(theta, X, y, l=1):
    '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term


def regularized_gradient(theta, X, y, l=1):
    '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


if __name__ == '__main__':
    path = "D:\git\Coursera-ML\johnwittenauer\data\ex2data2.txt"
    df = pd.read_csv(path, names=['test1', 'test2', 'accepted'])
    print(df.head())

    # sns.set(context="notebook", style="ticks", font_scale=1.5)
    sns.set(context="notebook", style="ticks", font_scale=1.5)
    sns.lmplot('test1', 'test2', hue='accepted', data=df,
               size=6,
               fit_reg=False,
               scatter_kws={"s": 50}
               )

    plt.title('Regularized Logistic Regression')
    # plt.show()

    x1 = np.array(df.test1)
    x2 = np.array(df.test2)
    data = feature_mapping(x1, x2, power=6)
    print(data.shape)
    print(data.head())
    theta = np.zeros(data.shape[1])
    X = feature_mapping(x1, x2, power=6, as_ndarray=True)
    print(X.shape)

    y = get_y(df)
    print(y.shape)
    print(regularized_cost(theta, X, y, l=1))

    print(regularized_gradient(theta, X, y))
    print('init cost = {}'.format(regularized_cost(theta, X, y)))
    print(X.shape,y.shape,theta.shape)
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
    print(res)
    final_theta = res.x
    print(final_theta.shape)
    y_pred = predict(X, final_theta)

    print(classification_report(y, y_pred))


