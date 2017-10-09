from sklearn import linear_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # path = os.getcwd() + '..\data\ex1data2.txt'
    path = "D:\git\Coursera-ML\johnwittenauer\data\ex1data1.txt"
    data2 = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    print(data2.head())
    print(data2.describe())

    # data2 = (data2 - data2.mean()) / data2.std()  # Size   Bedrooms Price 各自用mean std来计算
    # data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    # data2 = (data2 - data2.mean()) / (data2.max() - data2.min())

    # print(data2.head())
    # print(data2.describe())

    # add ones column
    data2.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data2.shape[1]
    X = data2.iloc[:, 0:cols - 1]
    y = data2.iloc[:, cols - 1:cols]

    # convert to matrices and initialize theta
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    # theta2 = np.matrix(np.array([0, 0, 0]))

    model = linear_model.LinearRegression()
    model.fit(X, y)

    x = np.array(X[:, 1].A1)
    f = model.predict(X).flatten()
    print("score:", model.score(X, y))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data2.Population, data2.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()
