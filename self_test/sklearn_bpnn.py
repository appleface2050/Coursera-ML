import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    # wine = pd.read_csv('D:\git\Coursera-ML\johnwittenauer\data\wine_data.csv',
    #                    names=["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
    #                           "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity",
    #                           "Hue", "OD280", "Proline"])

    wine = pd.read_csv('D:\git\Coursera-ML\johnwittenauer\data\wine_data.csv',
                       names=["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
                              "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                              "Color_intensity", "Hue", "OD280", "Proline"])
    print(wine.head())
    print(wine.describe().transpose())

    X = wine.drop('Cultivator', axis=1)
    y = wine["Cultivator"]
    print(wine.shape)
    # print(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(X_train.shape)
    print(X_test.shape)

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)  # StandardScaler----计算训练集的平均值和标准差，以便测试数据集使用相同的变换

    scaler.fit(X_train)

    # print("scaler.mean_:", scaler.mean_)
    # print("scaler.std_:", scaler.std_)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500, activation="logistic")
    print(mlp.fit(X_train, y_train))
    predictions = mlp.predict(X_test)

    print(confusion_matrix(y_test, predictions))

    print(classification_report(y_test, predictions))

    # print(mlp.coefs_)
    # print(mlp.intercepts_)
    # a = np.array(X.iloc[1])
    # b = y[1]
    # theta0 = mlp.coefs_[0]
    # theta1 = mlp.coefs_[1]
    # theta2 = mlp.coefs_[2]
    # theta3 = mlp.coefs_[3]
    # print("theta0.shape:", theta0.shape)
    # print("theta1.shape:", theta1.shape)
    # print("theta2.shape:", theta2.shape)
    # print("theta3.shape:", theta3.shape)

    a = np.array(X.iloc[59])
    a = a.reshape((1,13))
    print(mlp.predict(a))