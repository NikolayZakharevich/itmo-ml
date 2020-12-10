import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import zero_one_loss, accuracy_score
from math import log
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier

COLOR_POINT_N = '#FD32F7'
COLOR_POINT_P = '#052609'

COLOR_AREA_N = '#8C1554'
COLOR_AREA_P = '#2F621B'

BASE_CLASSIFIER_MAX_DEPTH = 3


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
    def __init__(self, base_clf):
        self.base_clf = base_clf
        self.clfs_ = []
        self.clf_weights = []
        self.sample_weight = []

    def fit_iter(self, X, y):
        n, m = X.shape
        sample_weight = self.sample_weight if len(self.sample_weight) > 0 else np.repeat(1 / n, n)

        clf = deepcopy(self.base_clf)
        clf.fit(X, y, sample_weight=sample_weight)

        y_pred = clf.predict(X)
        N_t = zero_one_loss(y, y_pred, sample_weight=sample_weight, normalize=False)
        b_t = 0.5 * log((1 - N_t) / N_t)
        sample_weight *= np.exp(-b_t * (y * y_pred))

        self.sample_weight = sample_weight / np.sum(sample_weight)
        self.clf_weights.append(b_t)
        self.clfs_.append(clf)

    def predict(self, X):
        n, m = X.shape
        answers = np.zeros(n)
        for clf, w in zip(self.clfs_, self.clf_weights):
            answers += w * clf.predict(X)
        return np.sign(answers)


def show_steps(X, y, filename):
    x_step = 0.01
    y_step = 0.01
    x_min, y_min = broaden_min(np.amin(X, 0), x_step, y_step)
    x_max, y_max = broaden_max(np.amax(X, 0), x_step, y_step)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x')
    plt.ylabel('y')

    x_all, y_all = np.meshgrid(np.arange(x_min, x_max, x_step), np.arange(y_min, y_max, y_step))

    points = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    clf = AdaBoost(DecisionTreeClassifier(max_depth=BASE_CLASSIFIER_MAX_DEPTH))
    for it in range(1, 56):
        clf.fit_iter(X, y)
        if it in points:
            z_all = np.array(clf.predict(np.c_[x_all.ravel(), y_all.ravel()])).reshape(x_all.shape)
            plt.pcolormesh(x_all, y_all, z_all, cmap=ListedColormap([COLOR_AREA_N, COLOR_AREA_P]))

            plt.title('Iteration #%d "%s"' % (it, filename))
            plt.scatter(*X[y == -1].T, color=COLOR_POINT_N, s=10)
            plt.scatter(*X[y == 1].T, color=COLOR_POINT_P, s=10)
            plt.show()


def show_graph(X, y, filename):
    iterations = []
    accuracies = []
    clf = AdaBoost(DecisionTreeClassifier(max_depth=BASE_CLASSIFIER_MAX_DEPTH))
    for iteration in range(1, 56):
        iterations.append(iteration)
        clf.fit_iter(X, y)
        accuracies.append(accuracy_score(y, clf.predict(X)))

    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy dependence on count "%s"' % filename)
    plt.plot(iterations, accuracies)
    plt.show()


def run(filename: str):
    X, y = read_dataset(filename)
    show_steps(X, y, filename)
    show_graph(X, y, filename)


if __name__ == '__main__':
    run('chips.csv')
    run('geyser.csv')
