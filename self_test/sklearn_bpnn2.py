# coding:utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import scipy.io as sio
import datetime


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y


if __name__ == '__main__':
    now = datetime.datetime.now()

    X, y = load_data('D:\git\Coursera-ML\johnwittenauer\data\ex3data1.mat', transpose=False)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)  # StandardScaler----计算训练集的平均值和标准差，以便测试数据集使用相同的变换
    scaler.fit(X_train)
    # print("scaler.mean_:", scaler.mean_)
    # print("scaler.std_:", scaler.std_)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(400, 400, 400, 400, 400, 400),
                        max_iter=500, activation="logistic")

    print(mlp.fit(X_train, y_train))
    predictions = mlp.predict(X_test)

    print(confusion_matrix(y_test, predictions))

    print(classification_report(y_test, predictions))

    print(datetime.datetime.now() - now)
