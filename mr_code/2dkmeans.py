# coding:utf-8

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.cluster import KMeans

import sys
sys.path.append('..')

from helper import kmeans as km

if __name__ == '__main__':
    mat = sio.loadmat('data/ex7data2.mat')
    data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    print(data2.head())

    # sns.set(context="notebook", style="white")
    # sns.lmplot('X1', 'X2', data=data2, fit_reg=False)
    # plt.show()

    sk_kmeans = KMeans(n_clusters=3)
    sk_kmeans.fit(data2)
    sk_C = sk_kmeans.predict(data2)
    data_with_c = km.combine_data_C(data2, sk_C)
    sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
    plt.show()