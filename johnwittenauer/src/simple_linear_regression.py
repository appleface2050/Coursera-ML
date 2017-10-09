# coding=utf-8
"""
http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.lib import computeCost, gradient_descent


path = os.getcwd() + "\data\ex1data1.txt"
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())
print(data.describe())

data.plot(kind='scatter', x="Population", y="Profit", figsize=(12, 8))


# plt.show()



# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)

print(data.head())

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
# theta = np.matrix(np.array([0,0]))
theta = np.matrix(np.zeros(X.shape[1]))
theta = theta.T
print(X.shape, theta.shape, y.shape)

error = computeCost(X, y, theta)
print("error:", error)

iters = 20000

g, cost, final_cost = gradient_descent(X, y, theta, 0.01, iters)
print(g)
print(final_cost)


fig, ax = plt.subplots(figsize=(12,8))
ax.plot(X[:,1], (g[0, 0] + (g[1, 0] * X[:,1])), 'r', label='Prediction')

ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


