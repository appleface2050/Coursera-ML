# coding=utf-8

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5,5,5,5,5), activation="relu", max_iter=10000, verbose=True)

X = [
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
]

y = [0, 0, 1, 1]

scaler = StandardScaler()
# X = scaler.transform(X)

clf.fit(X, y)
result = clf.predict(X)
print(result)

print("loss:", clf.loss_)
print([coef.shape for coef in clf.coefs_])
