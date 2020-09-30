import numpy as np

class_map = {}


def read_data(file):
    return np.genfromtxt(file, dtype=None, delimiter=',', names=True, encoding='utf-8')


def fill_spaces(data, classes):
    return data, classes


def normalize(data, classes):
    row_sums = data.sum(axis=1)
    new_matrix = data / row_sums[:, np.newaxis]
    return new_matrix, classes


def vectorize(data):
    res = []
    classes = []
    for i, x in enumerate(data):
        if x[-1] not in class_map:
            class_map[x[-1]] = len(class_map)

        row = []
        for val in x.tolist()[:-1]:
            row.append(val)
        res.append(row)
        classes.append(class_map[x[-1]])
    return np.array(res, dtype=int), np.array(classes)


def read_and_prepare_data(file):
    res = {}

    return to_list(*normalize(*fill_spaces(*vectorize(read_data(file)))))


def read_data(file):
    return np.genfromtxt(file, dtype=None, delimiter=',', names=True, encoding='utf-8')


def to_list(data, classes):
    res = []
    for i, x in enumerate(data):
        res.append(x.tolist() + [classes[i]])
    return res
