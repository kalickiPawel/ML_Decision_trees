import numpy as np
import pandas as pd
import sys

from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

from Node import Node

infogain = lambda p: entropy(np.sum(p, axis=1)) + entropy(np.sum(p, axis=0)) - entropy(p)

entropy = lambda p: -np.sum(p * np.log2(p + 1e-16))


def freq(X, prob=True):
    xi, ni = np.unique(X, return_counts=True)
    return xi, (ni if not prob else ni / len(X))


def freq2(X, Y, prob=True):
    xi = np.unique(X)
    yj = np.unique(Y)
    x_sparse = False
    y_sparse = False
    if sparse.issparse(X):
        x_sparse = True
        xi = np.array([True, False])
    if sparse.issparse(Y):
        y_sparse = True
        yj = np.array([True, False])

    nij = np.zeros((xi.size, yj.size))
    if y_sparse and x_sparse:
        joined = sparse.hstack((X.astype('int32'), Y.astype('int32')))
        df = pd.DataFrame(joined.todense())
        df["D"] = np.zeros([X.shape[0], 1])
        df.columns = ['X', 'Y', 'D']
        pt = pd.pivot_table(df, values=['D'], index=['X', 'Y'], aggfunc={'D': 'count'})
        flattened = pd.DataFrame(pt.to_records())
        for index, row in flattened.iterrows():
            nij[row['X'], row['Y']] = row['D']
    else:
        for i, x in enumerate(xi):
            for j, y in enumerate(yj):
                tested_and = np.logical_and(X == x, Y == y)
                nij[i, j] = np.sum(tested_and)

    if prob:
        nij = nij / np.sum(nij)

    return xi, yj, nij


def split_data(X, y, col_split):
    y_left, y_right = y[X[:, col_split] == 0], y[X[:, col_split] != 0]
    X_left, X_right = X[X[:, col_split] == 0, :], X[X[:, col_split] != 0, :]
    return X_left, y_left, X_right, y_right


class Tree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth):
        # super(Tree, self).__init__(arg)
        self.depth = 0
        self.max_depth = max_depth
        self.root = None
        self.gid = None

    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def all_same(self, items):
        # return all(x == items[0] for x in items)
        # print(items)
        return len(set(items)) == 1

    def fit(self, X, y):
        self.gid = 0
        self.root = Node(None, 0, self.gid)
        self.rpart(X, y, self.root)

    def rpart(self, X, y, parent, depth=0):
        print(f'depth: {depth}, gid: {self.gid}, parent.gid: {parent.gid}')
        feature_index, ig = self.find_best_split_of_all(X, y)
        yi, pi = freq(y, True)
        parent.yi = yi
        parent.pi = pi

        if depth >= self.max_depth:
            return
        elif ig == 0:
            return

        parent.feature_index = feature_index

        self.gid = self.gid + 1
        left = Node(parent, depth + 1, self.gid)
        self.gid = self.gid + 1
        right = Node(parent, depth + 1, self.gid)

        parent.left = left
        parent.right = right

        X_left, y_left, X_right, y_right = split_data(X, y, feature_index)

        self.rpart(X_left, y_left, left, depth + 1)
        self.rpart(X_right, y_right, right, depth + 1)

    def find_best_split_of_all(self, X, y):
        print("find_best_split_all")
        igain = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            xi, yj, pij = freq2(X[:, i], y)
            igain[i] = infogain(pij)

        col = np.argmax(igain)
        return col, igain[col]

    def get_node_list(self):
        nodes_list = []
        temp_list = [self.root]

        while len(temp_list) > 0:
            el = temp_list.pop()
            nodes_list.append(el)

            # nie teminalny
            if el.left is not None:
                temp_list.append(el.left)
            if el.right is not None:
                temp_list.append(el.right)

        return nodes_list

    def export_graph(self, file=sys.stdout, var_names=None):
        nodes = self.get_node_list()

        with file as f:
            print('graph G {', file=f)
            for n in nodes:
                if var_names is None:
                    print('{} [label="{}, {}, {}, {}"]'.format(
                        n.gid,
                        n.gid,
                        n.feature_index,
                        n.pi,
                        n.yi
                    ), file=f)
                else:
                    print('{} [label="{}, {}, {}, {}"]'.format(
                        n.gid,
                        n.gid,
                        var_names[n.feature_index],
                        n.pi,
                        n.yi
                    ), file=f)

            for n in nodes:
                if n.parent is not None:
                    if n.parent.left.gid == n.gid:
                        var = "False"
                    elif n.parent.right.gid == n.gid:
                        var = "True"
                    else:
                        var = ":D"

                    print('{}--{} [label="={}"]'.format(n.parent.gid, n.gid, var), file=f)
            print('}', file=f)

            if not f.closed:
                f.close()

        if file == sys.stdout:
            sys.stdout = open("/dev/stdout", "w")

    def predict(self, X):
        y = [None] * X.shape[0]
        print(len(y))

        for i in range(X.shape[0]):
            node = self.root
            while node.left is not None:
                node = node.left if not X[i, node.feature_index] else node.right
            y[i] = node.yi[np.argmax(node.pi)]

        return np.array(y)
