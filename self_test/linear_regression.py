# coding=utf-8
from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
from helper import linear_regression as lr  # my own module
from helper import general as general

data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

print(data.head())
# print (df.info())

sns.lmplot('population', 'profit', data, size=10, fit_reg=False)
# plt.show()
X = general.get_X(data)
print(X.shape, type(X))

y = general.get_y(data)
print(y.shape, type(y))

theta = np.zeros(X.shape[1])
print(theta.shape, type(theta))

print (lr.cost(theta, X, y))



epoch = 100000

min_max_scaler = preprocessing.MaxAbsScaler()
X= min_max_scaler.fit_transform(X)

final_theta, cost_data = lr.batch_gradient_decent(theta, X, y, epoch, 0.01)
print (cost_data[-1])

# theta_ne = lr.normal_equations(X, y)
# print (theta_ne)
# print (lr.cost(theta_ne, X, y))