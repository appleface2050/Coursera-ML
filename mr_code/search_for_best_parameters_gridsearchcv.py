# coding:utf-8

import numpy as np
import pandas as pd
from sklearn import svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import metrics
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    mat = sio.loadmat('data/ex6data3.mat')
    print(mat.keys())
    training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    training['y'] = mat.get('y')

    cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
    cv['y'] = mat.get('yval')


    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    parameters = {'C': candidate, 'gamma': candidate}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, n_jobs=4)
    clf.fit(training[['X1', 'X2']], training['y'])

    print(clf.best_params_)
    print(clf.best_score_)
    ypred = clf.predict(cv[['X1', 'X2']])
    print(metrics.classification_report(cv['y'], ypred))
