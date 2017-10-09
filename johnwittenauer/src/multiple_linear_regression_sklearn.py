from sklearn import linear_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "D:\git\Coursera-ML\johnwittenauer\data\ex1data2.txt"
    data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print(data2.head())
    print(data2.describe())

    # data2 = (data2 - data2.mean()) / data2.std()  # Size   Bedrooms Price 各自用mean std来计算
    # data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    # data2 = (data2 - data2.mean()) / (data2.max() - data2.min())

    print(data2.head())
    print(data2.describe())

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

    model = linear_model.LinearRegression()
    model.fit(X2, y2)

    x = np.array(X2[:, 1].A1)
    f = model.predict(X2).flatten()
    print(f)
    print("score:", model.score(X2, y2))
