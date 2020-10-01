from typing import List, Mapping, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from cf.b import solve as cf_b_solve
from cf.c import Dist, Kernel, Window, solve as cf_c_solve
from utils.data_helper import read_and_prepare_data

# https://www.openml.org/d/180
FILE_DATASET_TRAIN = '../datasets/dataset_184_covertype_200.csv'
FILE_DATASET_TEST = '../datasets/dataset_184_covertype_201_210.csv'


class HyperParams(NamedTuple):
    dist_type: Dist
    kernel_type: Kernel
    window_type: Window
    h_or_k: int


class Data(NamedTuple):
    d_train: List[List[float]]
    d_test: List[List[float]]
    classes: Mapping


def get_hyper_params_set() -> List[HyperParams]:
    result = []
    for dist_type in Dist:
        for kernel_type in Kernel:
            for window_type in Window:
                for h_or_k in [1, 3, 10, 15, 20, 50]:
                    result.append(HyperParams(dist_type, kernel_type, window_type, h_or_k))
    return result


def regression_reduction_naive(d_train: List[List[float]], hyper_params: HyperParams, q: List[float]) -> int:
    dist_type, kernel_type, window_type, h_or_k = hyper_params

    if window_type == Window.FIXED:
        h_or_k = h_or_k / 50

    return int(round(
        cf_c_solve(d_train, len(d_train), len(d_train[0]) - 1, q, h_or_k, dist_type, kernel_type, window_type)
    ))


def find_best_hyper_params(data: Data, hyper_params_set: List[HyperParams]):
    return max(hyper_params_set,
               key=lambda hp: sum(get_class(q) == regression_reduction_naive(data.d_train, hp, q) for q in data.d_test))


def calc_confusion_matrix(data: Data, hyper_params: HyperParams):
    sz = len(data.classes)
    matrix = np.zeros((sz, sz))
    for q in data.d_test:
        matrix[get_class(q)][regression_reduction_naive(data.d_train, hyper_params, q)] += 1
    return matrix


def show_f_measure_dependence(data: Data, hyper_params: HyperParams):
    window_width_all = []
    f_macro_all = []
    f_micro_all = []

    for h in np.arange(0, 50, 0.5):
        window_width_all.append(h)
        f_macro, f_micro = cf_b_solve(list(calc_confusion_matrix(data, hyper_params)), len(data.classes))
        f_macro_all.append(f_macro)
        f_micro_all.append(f_micro)

    plt.xlabel('Window width', fontsize=18)
    plt.ylabel('F-measure', fontsize=16)
    plt.title('F-measure dependence from window width')
    plt.plot(window_width_all, f_micro_all, f_macro_all)
    print(f_micro_all, f_macro_all)
    plt.show()


def get_class(row: list):
    return row[-1]


def run(file_dataset_train, file_dataset_test):
    d_train, classes = read_and_prepare_data(file_dataset_train)
    d_test, _ = read_and_prepare_data(file_dataset_test)
    data = Data(d_train, d_test, classes)

    # best_hyper_params = find_best_hyper_params(data, get_hyper_params_set())
    best_hyper_params = HyperParams(dist_type=Dist.MANHATTAN, kernel_type=Kernel.UNIFORM, window_type=Window.FIXED,
                                    h_or_k=3)

    print("Best hyper parameters: ", best_hyper_params)
    show_f_measure_dependence(data, best_hyper_params)


if __name__ == '__main__':
    run(FILE_DATASET_TRAIN, FILE_DATASET_TEST)
