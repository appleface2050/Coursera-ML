# coding:utf-8

from math import sqrt

import numpy as np
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# print (train.describe())
# print(train.columns)
# print(train.sample(5))

"""
Numerical Features: Age (Continuous), Fare (Continuous), SibSp (Discrete), Parch (Discrete)
Categorical Features: Survived, Sex, Embarked, Pclass
Alphanumeric Features: Ticket, Cabin
"""


# print(pd.isnull(train).sum())

# draw a bar plot of survival by sex
# sns.barplot(x="Sex", y="Survived", data=train)
def simplify_fare(df):
    df_new = df
    bins = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600]
    group_names = ["0+", "10+", "20+", "30+", "40+", "50+", "60+", "70+", "80+", "90+", "100+", "200+", "300+", "400+",
                   "500+"]
    categories = pd.cut(df_new.Fare, bins, labels=group_names)
    df_new['Fare_scope'] = categories
    return df_new


def simplify_ages(df):
    df_new = df
    bins = (0, 5, 13, 18, 25, 35, 60, 120)
    group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df_new.Age, bins, labels=group_names)
    df_new['Stage'] = categories
    return df_new


train = simplify_ages(train)
test = simplify_ages(test)

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

# sns.barplot(x="Sex", y="Survived", data=train)
# print (train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))
# sns.barplot(x="SibSp", y="Survived", data=train)
# sns.barplot(x="Parch", y="Survived", data=train)
# sns.barplot(x="Stage", y="Survived", data=train)
# sns.barplot(x="CabinBool", y="Survived", data=train)
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

train = simplify_fare(train)
test = simplify_fare(test)

# print(train[train["Fare_scope"].isnull()] )
#
# print(pd.isnull(train).sum())


combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

print(train.head())

plt.show()
