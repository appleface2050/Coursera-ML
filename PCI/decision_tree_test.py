# coding:utf-8

from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals.six import StringIO
# import pydotplus

data = []
labels = []


with open("data/car.data.txt") as ifile:
    for line in ifile:
        rowDict = {}  # data需要是字典形式，因为之后需要使用DictVectorizer()修改字符串数据类型，以便符合DecisionTreeClassifier()
        tokens = line.strip().split(',')
        rowDict['buying'] = tokens[0]  # 分割数据，将label与data分开
        rowDict['maint'] = tokens[1]
        rowDict['doors'] = tokens[2]
        rowDict['persons'] = tokens[3]
        rowDict['lug_boot'] = tokens[4]
        rowDict['safety'] = tokens[5]
        data.append(rowDict)
        labels.append(tokens[-1])



x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)#初始label全为0

print(x[1199])
print(labels[1199])
y[labels =='vgood']=1
y[labels =='good']=1
y[labels =='acc']=1
print(y[1199])

vec = DictVectorizer()#转换字符串数据类型
dx = vec.fit_transform(x).toarray()

# print(dx[:5])
# print(vec.get_feature_names())

clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=7,min_samples_split=20,min_samples_leaf=10)
clf = clf.fit(dx,y)#导入数据

with open("tree.dot",'w') as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file = f)#输出结果至文件
dot_data = StringIO()
tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=dot_data)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())