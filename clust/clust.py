import random
from enum import Enum
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

DATASET_FILENAME = 'dataset_191_wine.csv'

colors = ["b", "g", "r"]

N_CLASSES = 3

LABEL_IDX = 0
DELTA_STEPS = 20
INF = 10000000


class Dist(Enum):
    MANHATTAN = 'manhattan'
    EUCLIDEAN = 'euclidean'
    CHEBYSHEV = 'chebyshev'


DISTANCE_TYPE = Dist.EUCLIDEAN


#
# Read and show data:
#

def read_data():
    data = pd.read_csv(DATASET_FILENAME)
    data.head()
    Xs = data[data.columns[1:]]
    Xs.head()
    classes = np.unique(data.loc[:, data.columns[LABEL_IDX]])
    X_norm = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(Xs)
    y = data.loc[:, data.columns[LABEL_IDX]].apply(lambda status: np.where(classes == status)[0][0])
    return X_norm, y


def display_clusters(X_reduced, labels, title):
    plt.figure(figsize=(12, 12))
    unique_labels = np.unique(labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        cur_xs = X_reduced[labels == label, 0]
        cur_ys = X_reduced[labels == label, 1]
        plt.scatter(cur_xs, cur_ys, color=colors[i], alpha=0.5, label=label)
    plt.title(title)
    plt.xlabel("X координата")
    plt.ylabel("Y координата")
    plt.legend()
    plt.show()


#
# Algo state:
#

class State():
    distances: np.ndarray
    set_sizes: np.ndarray
    parent: List[int]
    rank: List[int]
    parents: Set[int]
    delta: float
    P: List[Tuple[int, int]]

    def __init__(self, X):
        n = len(X)
        distances = np.zeros((n, n))
        parent = [i for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = calc_distance(X[i], X[j])
                distances[i][j] = dist
                distances[j][i] = dist

        self.distances = distances
        self.set_sizes = np.ones(n, dtype=int)
        self.parent = parent
        self.rank = [0] * n
        self.parents = set(range(n))
        self.delta = INF
        self.P = []

    def calc_delta(self):
        delta = INF
        for i in range(DELTA_STEPS):
            u = random.choice(tuple(self.parents))
            v = random.choice(tuple(self.parents))
            if u < v:
                delta = min(delta, self.distances[u][v])
        self.delta = delta

    def calc_p(self):
        P = []
        for u in self.parents:
            for v in self.parents:
                if u < v and self.distances[u][v] <= self.delta:
                    P.append((u, v))
        self.P = P

    def next_iter(self):
        if len(self.P) == 0:
            self.calc_delta()
            self.calc_p()

        P_distances = []
        for u, v in self.P:
            u = self.find_set(u)
            v = self.find_set(v)
            if u != v:
                P_distances.append((u, v, self.distances[u][v]))

        P_distances.sort(key=lambda x : x[-1])
        print(self.delta, self.P)
        print(P_distances, end="\n\n")
        u, v, _dist = min(P_distances, key=lambda x: x[-1])
        self.union_sets(u, v)

    # DSU:

    def find_set(self, v: int):
        if v == self.parent[v]:
            return v
        res = self.find_set(self.parent[v])
        self.parent[v] = res
        return res

    def union_sets(self, u: int, v: int):
        u = self.find_set(u)
        v = self.find_set(v)
        if u != v:
            if self.rank[u] < self.rank[v]:
                u, v = v, u
            self.parent[v] = u
            if self.rank[u] == self.rank[v]:
                self.rank[u] += 1

        self.parents.remove(v)

        P = []
        for s in self.parents:
            if s == u:
                continue
            dist = distance_ward(self, u, v, s)
            self.distances[u][s] = dist
            self.distances[s][u] = dist
            if dist < self.delta:
                P.append((u, s))
        for (u1, v1) in self.P:
            u1 = self.find_set(u1)
            v1 = self.find_set(v1)
            if u1 != u and u1 != v and v1 != u and v1 != v:
                P.append((u1, v1))

        self.P = P
        self.set_sizes[u] += self.set_sizes[v]


#
# Distance:
#

def calc_distance(x: List[float], y: List[float]) -> float:
    if DISTANCE_TYPE == Dist.MANHATTAN:
        return sum(abs(x[i] - y[i]) for i in range(0, len(x)))
    elif DISTANCE_TYPE == Dist.EUCLIDEAN:
        return np.sqrt(sum((x[i] - y[i]) * (x[i] - y[i]) for i in range(0, len(x))))
    elif DISTANCE_TYPE == Dist.CHEBYSHEV:
        return max(abs(x[i] - y[i]) for i in range(0, len(x)))


def distance_ward(state: State, u, v, s) -> float:
    U = state.set_sizes[u]
    V = state.set_sizes[v]
    S = state.set_sizes[s]
    W = U + V
    aU = (S + U) / (S + W)
    aV = (S + V) / (S + W)
    beta = -S / (S + W)
    return aU * state.distances[u][s] + aV * state.distances[v][s] + beta * state.distances[u][v]


if __name__ == '__main__':
    X, y, = read_data()
    X_reduced = PCA(n_components=2).fit_transform(X)

    state = State(X)
    while len(state.parents) > N_CLASSES:
        state.next_iter()

    clusters = [state.find_set(x) for x in range(len(X))]
    classes, counts = np.unique(clusters, return_counts=True)
    print(counts)
    y_clustered = list(map(lambda status: np.where(classes == status)[0][0], clusters))

    display_clusters(X_reduced, y, "Настоящие метки")
    display_clusters(X_reduced, y_clustered, "Кластеризованные метки")