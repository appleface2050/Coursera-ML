# coding:utf-8
import math


class matchrow(object):
    def __init__(self, row, allnum=False):
        if allnum:
            self.data = [float(row[i]) for i in range(len(row) - 1)]
        else:
            self.data = row[0:len(row) - 1]
        self.match = int(row[len(row) - 1])

    def loadmatch(f, allnum=False):
        rows = []
        for line in open(f):
            rows.append(matchrow(line.split(','), allnum))
        return rows


def plotagematches(rows):
    import pylab
    xdm, ydm = [r.data[0] for r in rows if r.match == 1], [r.data[1] for r in rows if r.match == 1]
    xdn, ydn = [r.data[0] for r in rows if r.match == 0], [r.data[1] for r in rows if r.match == 0]

    pylab.plot(xdm, ydm, "go")
    pylab.plot(xdn, ydn, "ro")
    pylab.show()


def veclength(v):
    return sum([p ** 2 for p in v])


def rbf(v1, v2, gamma=10):
    dv = [v1[i] - v2[i] for i in range(len(v1))]
    l = veclength(dv)
    return math.e ** (-gamma * l)


def lineartrain(rows):
    averages = {}
    counts = {}

    for row in rows:
        # Get the class of this point
        cl = row.match

        averages.setdefault(cl, [0.0] * (len(row.data)))
        counts.setdefault(cl, 0)

        # Add this point to the averages
        for i in range(len(row.data)):
            averages[cl][i] += float(row.data[i])

        # Keep track of how many points in each class
        counts[cl] += 1

    # Divide sums by counts to get the averages
    for cl, avg in averages.items():
        for i in range(len(avg)):
            avg[i] /= counts[cl]
    return averages


def dotproduct(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])


def dpclassify(point, avgs):
    b = (dotproduct(avgs[1], avgs[1]) - dotproduct(avgs[0], avgs[0])) / 2
    y = dotproduct(point, avgs[0]) - dotproduct(point, avgs[1]) + b
    if y > 0:
        return 0
    else:
        return 1


def yesno(v):
    if v == 'yes':
        return 1
    elif v == 'no':
        return -1
    else:
        return 0


def matchcount(interest1, interest2):
    l1 = interest1.split(':')
    l2 = interest2.split(':')
    x = 0
    for v in l1:
        if v in l2: x += 1
    return x


if __name__ == '__main__':
    agesonly = matchrow.loadmatch('data/agesonly.csv', allnum=True)
    matchmaker = matchrow.loadmatch('data/matchmaker.csv')
    # plotagematches(agesonly)
    avgs = lineartrain(agesonly)
    print(avgs)
    print(dpclassify([30, 30], avgs))
