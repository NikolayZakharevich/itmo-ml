from typing import List, Mapping, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from cf.b import solve as cf_b_solve
from cf.c import Dist, Kernel, Window, solve as cf_c_solve
from utils.data_helper import read_and_prepare_data

# https://www.openml.org/d/180
#
# naive:   HyperParams(dist_type=<Dist.CHEBYSHEV: 'chebyshev'>, kernel_type=<Kernel.SIGMOID: 'sigmoid'>,
# window_type=<Window.VARIABLE: 'variable'>, h_or_k=1)
#
# one_hot: HyperParams(dist_type=<Dist.MANHATTAN: 'manhattan'>, kernel_type=<Kernel.TRIWEIGHT: 'triweight'>,
# window_type=<Window.FIXED: 'fixed'>, h_or_k=0.05)
FILE_DATASET_TRAIN_BAD = '../datasets/dataset_184_covertype_100.csv'

# https://www.openml.org/d/40476
#
# naive: Best hyper parameters:  HyperParams(dist_type=<Dist.MANHATTAN: 'manhattan'>, kernel_type=<Kernel.TRIANGULAR:
# 'triangular'>, window_type=<Window.FIXED: 'fixed'>, h_or_k=0.45)
#
# one_hot: Best hyper parameters:  HyperParams(dist_type=<Dist.MANHATTAN: 'manhattan'>,
# kernel_type=<Kernel.EPANECHNIKOV: 'epanechnikov'>, window_type=<Window.FIXED: 'fixed'>,  h_or_k=0.2)
FILE_DATASET_TRAIN_GOOD = '../datasets/dataset_thyroid-allhypo.csv'


class HyperParams(NamedTuple):
    dist_type: Dist
    kernel_type: Kernel
    window_type: Window
    h_or_k: int


class Data(NamedTuple):
    d_train: List[List[float]]
    classes: Mapping


def get_hyper_params_set(data: Data) -> List[HyperParams]:
    result = []
    for dist_type in Dist:
        for kernel_type in Kernel:
            for window_type in Window:
                if window_type == Window.FIXED:
                    h_or_k_all = np.arange(0, 1, 0.05)
                else:
                    h_or_k_all = range(1, len(data.d_train) - 1, 10)
                for h_or_k in h_or_k_all:
                    result.append(HyperParams(dist_type, kernel_type, window_type, h_or_k))
    return result


def one_hot_row_transformation(x: List[float], n_classes: int):
    res = x.copy()
    res.pop()
    x_class = get_class(x)
    for i in range(n_classes):
        res.append(1 if x_class == i else 0)
    return res


def regression_reduction_naive(d_train: List[List[float]], hyper_params: HyperParams, q: List[float]) -> int:
    dist_type, kernel_type, window_type, h_or_k = hyper_params
    return int(round(
        cf_c_solve(d_train, len(d_train), len(d_train[0]) - 1, q, h_or_k, dist_type, kernel_type, window_type)[0]
    ))


def regression_reduction_one_hot(d_train: List[List[float]], hyper_params: HyperParams, q: List[float],
                                 n_classes: int) -> int:
    dist_type, kernel_type, window_type, h_or_k = hyper_params
    m = len(d_train[0]) - n_classes

    regression_res = cf_c_solve(d_train, len(d_train), m, q, h_or_k, dist_type, kernel_type, window_type, n_classes)
    return max(range(0, n_classes), key=lambda idx: regression_res[idx])


def find_best_hyper_params(data: Data, hyper_params_set: List[HyperParams], use_one_hot: bool = True):
    return max(hyper_params_set,
               key=lambda hp: cf_b_solve(list(calc_confusion_matrix(data, hp, use_one_hot)), len(data.classes))[0])


def calc_confusion_matrix(data: Data, hyper_params: HyperParams, use_one_hot: bool = True):
    sz = len(data.classes)
    matrix = np.zeros((sz, sz))

    for idx_q, q in enumerate(data.d_train):
        if use_one_hot:
            d_train = [one_hot_row_transformation(x, sz) for idx_x, x in enumerate(data.d_train) if idx_x != idx_q]
            q_predicted_class = regression_reduction_one_hot(d_train, hyper_params, q, sz)
        else:
            d_train = [x for idx_x, x in enumerate(data.d_train) if idx_x != idx_q]
            q_predicted_class = regression_reduction_naive(d_train, hyper_params, q)
        matrix[get_class(q)][q_predicted_class] += 1
    print(hyper_params)
    return matrix


def show_f_measure_dependence(data: Data, hyper_params: HyperParams, use_one_hot: bool = True):
    f_macro_all = []
    f_micro_all = []

    if hyper_params.window_type == Window.FIXED:
        h_or_k_all = np.arange(0, 1, 0.05)
        h_or_k_label = 'window width'
    else:
        h_or_k_all = range(0, 10, 1)
        h_or_k_label = 'number of neighbours'

    for h_or_k in h_or_k_all:
        hp = HyperParams(hyper_params.dist_type, hyper_params.kernel_type, hyper_params.window_type, h_or_k)
        f_macro, f_micro = cf_b_solve(list(calc_confusion_matrix(data, hp, use_one_hot)), len(data.classes))
        print(h_or_k, f_macro, f_micro)
        f_macro_all.append(f_macro)
        f_micro_all.append(f_micro)

    plt.xlabel(h_or_k_label, fontsize=16)
    plt.ylabel('F-measure', fontsize=16)
    plt.title('F-measure dependence from ' + h_or_k_label)
    plt.plot(h_or_k_all, f_macro_all, label='F-macro')
    plt.plot(h_or_k_all, f_micro_all, label='F-micro')
    plt.legend()
    plt.show()


def get_class(row: list):
    return row[-1]


def run(file_dataset_train, use_one_hot: bool = True):
    d_train, classes = read_and_prepare_data(file_dataset_train)
    data = Data(d_train, classes)

    best_hyper_params = find_best_hyper_params(data, get_hyper_params_set(data), use_one_hot)
    print("Best hyper parameters: ", best_hyper_params)

    show_f_measure_dependence(data, best_hyper_params, use_one_hot)


if __name__ == '__main__':
    run(FILE_DATASET_TRAIN_GOOD, False)
