# coding:utf-8

import numpy as np
np.random.seed(1337)  # for reproducibility

import datetime

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn import decomposition


(X_train, y_train), (X_test, y_test) = mnist.load_data()

#PCA
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')

X = np.concatenate((X_train,X_test), axis=0)

pca = decomposition.PCA(n_components=196)
pca.fit(X)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print (X_train.shape)
print (X_test.shape)

# data pre-processing
X_train = X_train.reshape(-1, 1,14, 14)/255.
X_test = X_test.reshape(-1, 1,14, 14)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)





# Another way to build your CNN
model = Sequential()

start = datetime.datetime.now()

# Conv layer 1 output shape (32, 28, 28)
model.add(Conv2D(
    batch_input_shape=(None, 1, 14, 14),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,              #相当于(2,2)
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Conv2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
# model.add(Dense(1024))
model.add(Dense(200))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)



# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)


print ("used time", datetime.datetime.now() - start)