# coding=utf-8

import numpy as np
from sklearn import preprocessing

if __name__ == '__main__':
    a = np.array([
        [13, 2, 4, 41, 27],
        [12, 4, 3, 46, 21],
        [10, 5, 4, 29, 20],
        [10, 4, 4, 31, 18],
        [9, 4, 6, 27, 24],
        [8, 6, 5, 36, 29],
        [8, 5, 6, 31, 25],
        [7, 6, 6, 27, 25],
        [6, 5, 7, 35, 29],
        [6, 4, 9, 24, 29],
        [5, 6, 7, 20, 26],
        [4, 7, 8, 20, 27],
        [3, 8, 8, 22, 28],
        [4, 4, 11, 21, 43],
        [3, 6, 9, 12, 26],
        [2, 4, 13, 14, 39]]
    )
    min_max_scaler = preprocessing.MinMaxScaler()
    # print(a)
    print (min_max_scaler.fit_transform(a))