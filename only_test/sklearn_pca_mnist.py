from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition

np.random.seed(5)

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# X = np.concatenate((x_train, x_test), axis=0)

plt.cla()
pca = decomposition.PCA(n_components=200)
pca.fit(X)
X = pca.transform(X)
print (X.shape)





