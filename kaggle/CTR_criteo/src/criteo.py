# coding:utf-8


# https://www.kaggle.com/c/criteo-display-ad-challenge/

from __future__ import print_function

from sklearn import preprocessing
import numpy as np
import pandas

np.random.seed(1337)

# enc = preprocessing.OneHotEncoder()
# enc.fit([[0, 0, 3],
#          [1, 1, 0],
#          [0, 2, 1],
#          [1, 0, 2]])
# print(enc.n_values_)
# print(enc.feature_indices_)
# print(enc.transform([[0, 0, 0],
#                      [0, 2, 1],
#                      ]).toarray())

# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop
# import datetime



# a = np.loadtxt('../data/small.txt')
# print(a.shape)

df = pandas.read_csv('../data/small.txt', sep='\t', header=None)
# print(df.head(10))
print(df.loc[1])

# batch_size = 128
# num_classes = 10
# epochs = 8
#
# # the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# now = datetime.datetime.now()
#
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# model = Sequential()
# # model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu', input_dim=784))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               # optimizer=RMSprop(),
#                 optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test)
#                     )
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# print("use time:", datetime.datetime.now()-now)
