# coding:utf-8

import datetime
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import scipy.io as sio

if __name__ == '__main__':
    mat_tr = sio.loadmat('data/spamTrain.mat')
    print(mat_tr.keys())
    X, y = mat_tr.get('X'), mat_tr.get('y').ravel()
    print(X.shape, y.shape)

    mat_test = sio.loadmat('data/spamTest.mat')
    print(mat_test.keys())
    test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()
    print(test_X.shape, test_y.shape)
    # svc = svm.SVC()
    now = datetime.datetime.now()
    # svc.fit(X, y)

    # pred = svc.predict(test_X)
    # print(metrics.classification_report(test_y, pred))

    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    parameters = {'C': candidate, 'gamma': candidate}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, n_jobs=4)
    clf.fit(X, y)
    print(datetime.datetime.now() - now)

    print(clf.best_params_)
    print(clf.best_score_)
    ypred = clf.predict(test_X)
    print(metrics.classification_report(test_y, ypred))

    # logit_clf = LogisticRegression()
    # now = datetime.datetime.now()
    # logit_clf.fit(X, y)
    # print(datetime.datetime.now() - now)
    # pred = logit_clf.predict(test_X)
    # print(metrics.classification_report(test_y, pred))