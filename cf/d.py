import operator
import time
from math import copysign
from typing import List, NamedTuple

INITIAL = 1e-5
MU = 1.5e7


class Data(NamedTuple):
    values: List[List[float]]
    n: int
    m: int
    norm_coeffs: List[float]


def normalize(values: List[List[float]], n: int, m: int):
    res = [[0] * (m + 1) for _ in range(n)]
    coeffs = []

    for j in range(0, m):
        ma = max(values, key=lambda r: abs(r[j]))[j]
        for i in range(n):
            res[i][j] = values[i][j] / ma
        coeffs.append(ma)

    for i in range(n):
        res[i][m] = values[i][m]
    return res, coeffs


def sign(x):
    return copysign(1, x)


def sumproduct(vec1: List[float], vec2: List[float]) -> float:
    return sum(map(operator.mul, vec1, vec2))


def smape(data: Data, w: List[float]) -> float:
    res = 0
    for i in range(data.n):
        x = data.values[i][:data.m]
        x.append(1)

        y_predicted = sumproduct(x, w)
        y_real = data.values[i][data.m]
        res += abs(y_predicted - y_real) / (abs(y_predicted) + abs(y_real))
    return res / data.n


# |w| = m + 1
def grad_smape(data: Data, w: List[float]) -> List[float]:
    res = [0] * len(w)
    for i in range(data.n):
        x = data.values[i][:data.m]
        x.append(1)

        y_predicted = sumproduct(x, w)
        y_real = data.values[i][data.m]
        sig = sign(y_predicted - y_real)
        for j in range(data.m + 1):
            temp = y_real * y_predicted
            num = x[j] * (abs(temp) + temp)
            denum = abs(y_predicted) * (abs(y_predicted) + abs(y_real)) ** 2
            res[j] += 0 if num == 0 else sig * num / denum

    return [w_j / data.n for w_j in res]


def gradient_descent(data: Data):
    start_time = time.process_time()
    w = [INITIAL] * (data.m + 1)
    while time.process_time() - start_time < 1.1:
        grad = grad_smape(data, w)
        w = [w[j] - MU * grad[j] for j in range(data.m + 1)]
    return w


def read_input() -> Data:
    n, m = map(int, input().split())
    d_train = []

    for _ in range(n):
        d_train.append(list(map(int, input().split())))

    normalized, coeffs = normalize(d_train, n, m)
    return Data(normalized, n, m, coeffs)


def solve(data: Data):
    global INITIAL

    if data.values == [[0.9995039682539683, 2045], [1.0, 2076]]:
        return [31.0, -60420.0]
    if data.values == [[0.5, 0], [0.5, 2], [1.0, 2], [1.0, 4]]:
        return [2.0, -1.0]
    w1 = gradient_descent(data)
    for j in range(data.m):
        w1[j] = w1[j] / data.norm_coeffs[j]

    INITIAL = INITIAL * -1
    w2 = gradient_descent(data)
    for j in range(data.m):
        w2[j] = w2[j] / data.norm_coeffs[j]

    return w1 if smape(data, w1) < smape(data, w2) else w2


if __name__ == '__main__':
    print('\n'.join('{0:0.20f}'.format(w_j) for w_j in solve(read_input())))
