import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import zero_one_loss, accuracy_score
from math import log, isclose
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier

COLOR_POINT_N = '#FD32F7'
COLOR_POINT_P = '#052609'

COLOR_AREA_N = '#8C1554'
COLOR_AREA_P = '#2F621B'


def read_dataset(filename):
    values = pd.read_csv(filename).values
    np.random.shuffle(values)
    X = values[:, :-1]
    Y = np.vectorize(lambda t: 1 if t == 'P' else -1)(values[:, -1])
    return X, Y


def broaden_min(bounds, dx, dy):
    return bounds[0] - dx, bounds[1] - dy


def broaden_max(bounds, dx, dy):
    return bounds[0] + dx, bounds[1] + dy


class AdaBoost(BaseEstimator):
    def __init__(self, base_clf, iteration_cnt=0):
        self.base_clf = base_clf
        self.iteration_cnt = iteration_cnt
        self._clear()

    def fit_iter(self, X, y):
        if len(self.clf_weights) > 0 and (self.clf_weights[-1] == 1 or self.clf_weights[-1] == -1):
            return

        n, m = X.shape
        sample_weight = self.sample_weight if len(self.sample_weight) > 0 else np.repeat(1 / n, n)

        clf = deepcopy(self.base_clf)
        clf.fit(X, y, sample_weight=sample_weight)
        y_pred = clf.predict(X)
        error = zero_one_loss(y, y_pred, normalize=False, sample_weight=sample_weight)

        alpha = 0.5 * log((1 - error) / error)
        self.clf_weights.append(alpha)
        sample_weight *= np.exp(-alpha * (y * y_pred))
        self.sample_weight = sample_weight / np.sum(sample_weight)

        self.clfs_.append(clf)

    def fit(self, X, y):
        self._clear()
        for _ in range(self.iteration_cnt):
            self.fit_iter(X, y)

    def predict(self, X):
        n, m = X.shape
        answers = np.zeros(n)
        for clf, w in zip(self.clfs_, self.clf_weights):
            answers += w * clf.predict(X)
        return np.sign(answers)

    def _try_end_fit(self, result, error, clf):
        if isclose(error, 0.5 * result + 0.5):
            self.clfs_ = [clf]
            self.clf_weights = [result]

    def _clear(self):
        self.clfs_ = []
        self.clf_weights = []
        self.sample_weight = []


def show_steps(X, y, step_x, step_y=0.01):
    x_min, y_min = broaden_min(np.amin(X, 0), step_x, step_y)
    x_max, y_max = broaden_max(np.amax(X, 0), step_x, step_y)
    plt.figure(figsize=(10, 10))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x), np.arange(y_min, y_max, step_y))

    points = [55]
    # points = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    clf = AdaBoost(DecisionTreeClassifier(max_depth=1))
    for iter in range(56):
        clf.fit_iter(X, y)
        if iter in points:
            zz = np.array(clf.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
            plt.pcolormesh(xx, yy, zz, cmap=ListedColormap([COLOR_AREA_N, COLOR_AREA_P]))

            plt.scatter(*X[y == -1].T, color=COLOR_POINT_N, s=100)
            plt.scatter(*X[y == 1].T, color=COLOR_POINT_P, s=100)
            plt.show()


def show_graph(X, y):
    iterations = []
    accuracies = []
    clf = AdaBoost(DecisionTreeClassifier(max_depth=5))
    for iteration in range(1, 101):
        iterations.append(iteration)
        clf.fit_iter(X, y)
        accuracies.append(accuracy_score(y, clf.predict(X)))

    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy dependence on count')
    plt.plot(iterations, accuracies)
    plt.show()


def run(filename: str, draw_step: float):
    X, y = read_dataset(filename)
    show_steps(X, y, draw_step)
    show_graph(X, y)


if __name__ == '__main__':
    run('chips.csv', 0.01)
    run('geyser.csv', 1)
