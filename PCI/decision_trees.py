# coding:utf-8

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


class DecisionNode(object):
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.tb = tb
        self.fb = fb
        self.results = results

    # 在某一列上对数据集合进行拆分，能够处理数值型数据或名词性数据
    @classmethod
    def divideset(cls, rows, column, value):
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        else:
            split_function = lambda row: row[column] == value

        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]
        return (set1, set2)

    @classmethod
    def uniquecounts(cls, rows):
        results = {}
        for row in rows:
            r = row[len(row) - 1]
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    @classmethod
    def giniimpurity(cls, rows):
        total = len(rows)
        counts = cls.uniquecounts(rows)
        imp = 0
        for k1 in counts:
            p1 = float(counts[k1]) / total
            for k2 in counts:
                if k1 == k2:
                    continue
                p2 = float(counts[k2]) / total
                imp += p1 * p2
        return imp

    def printtree(tree, indent=''):
        # Is this a leaf node?
        if tree.results != None:
            print(str(tree.results))
        else:
            # Print the criteria
            print(str(tree.col) + ':' + str(tree.value) + '? ')

            # Print the branches
            print(indent + 'T->')
            DecisionNode.printtree(tree.tb, indent + '  ')
            print(indent + 'F->')
            DecisionNode.printtree(tree.fb, indent + '  ')

    def entropy(rows):
        from math import log
        log2 = lambda x: log(x) / log(2)
        results = DecisionNode.uniquecounts(rows)

        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            ent = ent - p * log2(p)
        return ent

    @classmethod
    def buildtree(cls, rows, scoref=entropy):
        if len(rows) == 0:
            return cls()
        current_score = scoref(rows)

        # Set up some variables to track the best criteria

        best_gain = 0.0
        best_criteria = None
        best_sets = None

        column_count = len(rows[0]) - 1
        for col in range(0, column_count):
            column_values = {}
            for row in rows:
                column_values[row[col]] = 1
            for value in column_values.keys():
                (set1, set2) = cls.divideset(rows, col, value)

            # Information gain
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
        if best_gain > 0:
            trueBranch = cls.buildtree(best_sets[0])
            falseBranch = cls.buildtree(best_sets[1])
            return cls(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
        else:
            return cls(results=cls.uniquecounts(rows))


if __name__ == '__main__':
    print(DecisionNode.divideset(my_data, 2, 'yes'))
    print(DecisionNode.entropy(my_data))

    tree = DecisionNode.buildtree(my_data)
    print(DecisionNode.printtree(tree))
