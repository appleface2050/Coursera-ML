# coding:utf-8

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn import tree

my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]

df = pd.DataFrame(my_data)
df.columns = ["refer", "location", "FAQ", "pv", "buy"]
print(df)

# print (df.ix[[0]])
# print("++++++++++++++")
# print (type(df[0]))
# print (df[0])
# print("++++++++++++++")
# print (type(df[[0]]))
# print (df[[0]])
# print("=============")
# print(type(df.ix[[0, 2, 4, 5, 7]]))
# print(df.ix[[0, 2, 4, 5, 7],[0, 1]])
# print(df.ix[:,[1,2]])

clf = DecisionTreeClassifier()
X = df[['refer', 'location', 'FAQ', 'pv']]
Y = df[['buy']]

# print(Y)
le_buy = preprocessing.LabelEncoder()
le_buy = le_buy.fit(Y['buy'])
# print(le_buy.classes_)
# Y['buy'] = le_buy.transform(Y['buy'])

Y.loc[:, 'buy'] = le_buy.transform(Y['buy'])

le_refer = preprocessing.LabelEncoder()
le_location = preprocessing.LabelEncoder()
le_FAQ = preprocessing.LabelEncoder()

le_refer = le_refer.fit(X['refer'])
le_location = le_location.fit(X['location'])
le_FAQ = le_FAQ.fit(X['FAQ'])

X.loc[:, "refer"] = le_refer.transform(X['refer'])
X.loc[:, "location"] = le_location.transform(X['location'])
X.loc[:, "FAQ"] = le_FAQ.transform(X['FAQ'])

clf = clf.fit(X, Y)

predictions = clf.predict(X)

# print (clf.predict([[4,3,1,18]]))
print(predictions, le_buy.inverse_transform(predictions))
from sklearn.metrics import make_scorer, accuracy_score
print(accuracy_score(Y['buy'], predictions))
