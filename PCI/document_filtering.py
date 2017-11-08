# coding:utf-8
import re
import math


def getwords(doc):
    splitter = re.compile('\\W*')

    # Split the words by non-alpha characters
    words = [s.lower() for s in splitter.split(doc)
             if len(s) > 2 and len(s) < 20]

    # Return the unique set of words only
    return dict([(w, 1) for w in words])


class classifier:
    def __init__(self, getfeatures, filename=None):
        # Counts of feature/category combinations
        self.fc = {}
        # Counts of documents in each category
        self.cc = {}
        self.getfeatures = getfeatures

    # 增加对特征/分类组合的计数值
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    def incc(self, cat):
        # if f in self.fc and cat in self.fc[f]:
        #     return float(self.fc[f][cat])
        # return 0.0
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    def totalcount(self):
        return sum(self.cc.values())

    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.getfeatures(item)
        for f in features:
            self.incf(f, cat)
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0
        return self.fcount(f, cat) / self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        # 计算当前概率值
        basicprob = prf(f, cat)

        # 统计特征在所有分类中出现的次数
        totals = sum([self.fcount(f, c) for c in self.categories()])

        # 计算加权平均
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp

    @classmethod
    def sampletrain(cls, cl):
        cl.train("Nobody owns the water.", "good")
        cl.train("the quick rabbit jumps fences", "good")
        cl.train("buy pharmaceuticals now", "bad")
        cl.train('make quick money in the online casino', 'bad')
        cl.train('the quick brown fox jumps', "good")


class naviebayes(classifier):
    """
    贝叶斯定理： P(A/B) * P(B) = P(B/A) * P(A)
    """

    def docprob(self, item, cat):
        """
        计算 P(Document| Category)
        """
        features = self.getfeatures(item)
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        catprob = self.catcount(cat)/self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob

if __name__ == '__main__':
    # cl = classifier(getwords)
    # print(cl.fcount('quick', 'good'))
    # print(cl.fcount('quick', 'bad'))
    # classifier.sampletrain(cl)
    # print(cl.weightedprob('money', 'good', cl.fprob))
    # # print(cl.fprob('quick', "good"))
    #
    # classifier.sampletrain(cl)
    # print(cl.weightedprob('money', 'good', cl.fprob))

    cl = naviebayes(getwords)
    classifier.sampletrain(cl)
    print (cl.prob('quick', 'bad'))


