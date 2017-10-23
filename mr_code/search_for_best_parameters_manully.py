# coding:utf-8

import numpy as np
import pandas as pd
from sklearn import svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import metrics

if __name__ == '__main__':
    mat = sio.loadmat('data/ex6data3.mat')
    print(mat.keys())
    training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    training['y'] = mat.get('y')

    cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
    cv['y'] = mat.get('yval')
    print(training.shape)
    print(training.head())

    print(cv.shape)
    print(cv.head())

    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    # gamma to comply with sklearn parameter name
    combination = [(C, gamma) for C in candidate for gamma in candidate]
    print(len(combination))

    search = []

    for C, gamma in combination:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(training[['X1', 'X2']], training['y'])
        search.append(svc.score(cv[['X1', 'X2']], cv['y']))
    print(search)
    best_score = search[np.argmax(search)]
    best_param = combination[np.argmax(search)]
    print(best_score, best_param)

    best_svc = svm.SVC(C=best_param[1], gamma=best_param[0])
    best_svc.fit(training[['X1', 'X2']], training['y'])
    ypred = best_svc.predict(cv[['X1', 'X2']])

    print(metrics.classification_report(cv['y'], ypred))