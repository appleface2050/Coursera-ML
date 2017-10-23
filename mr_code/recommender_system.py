# coding:utf-8

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))

import numpy as np
import pandas as pd
import scipy.io as sio

import sys
sys.path.append('..')

from helper import recommender as rcmd

if __name__ == '__main__':
    movies_mat = sio.loadmat('data/ex8_movies.mat')
    Y, R = movies_mat.get('Y'), movies_mat.get('R')
    print(Y.shape, R.shape)

    m, u = Y.shape
    # m: how many movies
    # u: how many users

    n = 10  # how many features for a movie

    param_mat = sio.loadmat('data/ex8_movieParams.mat')
    theta, X = param_mat.get('Theta'), param_mat.get('X')

    print(theta.shape, X.shape)




