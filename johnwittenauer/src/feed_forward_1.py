import numpy as np

import sys

sys.path.append('..')

from helper import nn
from helper import logistic_regression as lr

from sklearn.metrics import classification_report


if __name__ == '__main__':
    theta1, theta2 = nn.load_weight('D:\git\Coursera-ML\johnwittenauer\data\ex3weights.mat')
    print(theta1.shape, theta2.shape)

    X, y = nn.load_data('D:\git\Coursera-ML\johnwittenauer\data\ex3data1.mat', transpose=False)

    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept

    print(X.shape, y.shape)
    # feed forward prediction

    a1 = X
    z2 = a1 @ theta1.T # (5000, 401) @ (25,401).T = (5000, 25)
    print(z2.shape)
    z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
    a2 = lr.sigmoid(z2)
    print(a2.shape)

    z3 = a2 @ theta2.T
    print(z3.shape)
    a3 = lr.sigmoid(z3)
    print(a3.shape)
    y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention
    print(y_pred.shape)
    print(classification_report(y_pred, y_pred))



