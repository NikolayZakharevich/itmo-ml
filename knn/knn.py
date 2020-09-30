from typing import List, NamedTuple

import matplotlib.pyplot as plt

from cf.c import Dist, Kernel, Window, solve
from utils.data_helper import read_data, vectorize, fill_spaces, normalize, to_list

# https://www.openml.org/d/180
FILE_DATASET_TRAIN = '../datasets/dataset_184_covertype_200.csv'
FILE_DATASET_TEST = '../datasets/dataset_184_covertype_201_210.csv'


class HyperParams(NamedTuple):
    dist_type: Dist
    kernel_type: Kernel
    window_type: Window
    h_or_k: int


def read_and_prepare_data(file):
    return to_list(*normalize(*fill_spaces(*vectorize(read_data(file)))))


def get_hyper_params_set() -> List[HyperParams]:
    result = []
    for dist_type in Dist:
        for kernel_type in Kernel:
            for window_type in Window:
                for h_or_k in [1, 3, 10, 15, 20, 50]:
                    result.append(HyperParams(dist_type, kernel_type, window_type, h_or_k))
    return result


def knn_naive(d_train, hyper_params: HyperParams, q):
    dist_type, kernel_type, window_type, h_or_k = hyper_params

    if window_type == Window.FIXED:
        h_or_k = h_or_k / 50

    return round(
        solve(d_train, len(d_train), len(d_train[0]) - 1, q, h_or_k, dist_type, kernel_type, window_type)
    )


def get_class(row):
    return row[-1]


def show_hyper_params(good: list, n: int):
    plt.xlabel('hyper-parameters index', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    plt.title('Search for best hyper-parameters set')

    x_all = []
    y_all = []

    for idx, cnt in enumerate(good):
        x_all.append(idx)
        y_all.append(cnt / n)

    print(x_all)
    print(len(x_all))
    print(y_all)
    print(len(y_all))
    plt.plot(x_all, y_all)
    # plt.plot(x_all, y_all, 'r.', ms=2)
    plt.show()
    print("HERE")


def run(file_dataset_train, file_dataset_test):
    d_train = read_and_prepare_data(file_dataset_train)
    d_test = read_and_prepare_data(file_dataset_test)

    best_hyper_params = max(get_hyper_params_set(),
                            key=lambda hp: sum(get_class(q) == knn_naive(d_train, hp, q) for q in d_test))
    print(best_hyper_params)


if __name__ == '__main__':
    run(FILE_DATASET_TRAIN, FILE_DATASET_TEST)
