# coding:utf-8

import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create some data

X = np.linspace(-1, 1, 200)
# X = np.linspace(1, 100, 100)

np.random.shuffle(X)  # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, 200)

# Y = 0.5 * X + 2 + np.random.normal(0, 1, (100, )) #生成符合正态分布的数据集，mean=0 标准差为5 ， 1维 , 500条数据

# plot data


plt.scatter(X, Y)
plt.show()

a = int(200 * 0.8)
X_train, Y_train = X[:a], Y[:a]  # first 160 data points
X_test, Y_test = X[a:], Y[a:]  # last 40 data points

# build a neural network from the 1st layer to the last layer
model = Sequential()

model.add(Dense(units=1, input_dim=1))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# training
# print('Training -----------')
# for step in range(1000):
#     cost = model.train_on_batch(X_train, Y_train)
#     print('train cost: ', cost)
#     if step % 1000 == 0:
#         print('train cost: ', cost)
# training
model.fit(X_train, Y_train, epochs=300, batch_size=20)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()

# json_string = model.to_json()
# print(json_string)
# model.save_weights('my_model_weights.h5')

model.summary()



