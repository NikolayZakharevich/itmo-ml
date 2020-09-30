import math
from enum import Enum


class Dist(Enum):
    MANHATTAN = 'manhattan'
    EUCLIDEAN = 'euclidean'
    CHEBYSHEV = 'chebyshev'


class Kernel(Enum):
    UNIFORM = 'uniform'
    TRIANGULAR = 'triangular'
    EPANECHNIKOV = 'epanechnikov'
    QUARTIC = 'quartic'
    TRIWEIGHT = 'triweight'
    TRICUBE = 'tricube'
    GAUSSIAN = 'gaussian'
    COSINE = 'cosine'
    LOGISTIC = 'logistic'
    SIGMOID = 'sigmoid'


class Window(Enum):
    FIXED = 'fixed'
    VARIABLE = 'variable'


CONST_PI = 3.14159265358979323846264338327950288
CONST_15_DIV_16 = 0.9375
CONST_35_DIV_32 = 35.0 / 32.0
CONST_70_DIV_81 = 70.0 / 81.0
CONST_1_DIV_SQRT_2_PI = 1 / math.sqrt(2 * CONST_PI)
CONST_PI_DIV_4 = CONST_PI / 4
CONST_PI_DIV_2 = CONST_PI / 2
CONST_2_DIV_PI = 2 / CONST_PI


def calc_distance(d: Dist, x: list, y: list, m: int) -> float:
    if d == Dist.MANHATTAN:
        su = 0
        for i in range(0, m):
            su += abs(x[i] - y[i])
        return su
    elif d == Dist.EUCLIDEAN:
        su = 0.0
        for i in range(0, m):
            diff = x[i] - y[i]
            su += diff * diff
        return math.sqrt(su)
    elif d == Dist.CHEBYSHEV:
        ma = 0
        for i in range(0, m):
            ma = max(ma, abs(x[i] - y[i]))
        return ma


def calc_kernel(k: Kernel, d: float, h: float) -> float:
    u = d / h
    is_supported = abs(u) < 1
    if k == Kernel.UNIFORM:
        return 0.5 if is_supported else 0
    elif k == Kernel.TRIANGULAR:
        return 1 - abs(u) if is_supported else 0
    elif k == Kernel.EPANECHNIKOV:
        return 0.75 * (1 - u * u) if is_supported else 0
    elif k == Kernel.QUARTIC:
        if not is_supported:
            return 0
        temp = 1 - u * u
        return CONST_15_DIV_16 * temp * temp
    elif k == Kernel.TRIWEIGHT:
        if not is_supported:
            return 0
        temp = 1 - u * u
        return CONST_35_DIV_32 * temp * temp * temp
    elif k == Kernel.TRICUBE:
        if not is_supported:
            return 0
        temp = 1 - abs(u * u * u)
        return CONST_70_DIV_81 * temp * temp * temp
    elif k == Kernel.GAUSSIAN:
        return CONST_1_DIV_SQRT_2_PI * math.exp(-0.5 * u * u)
    elif k == Kernel.COSINE:
        return CONST_PI_DIV_4 * math.cos(CONST_PI_DIV_2 * u) if is_supported else 0
    elif k == Kernel.LOGISTIC:
        return 1 / (math.exp(u) + 2 + math.exp(-u))
    elif k == Kernel.SIGMOID:
        return CONST_2_DIV_PI / (math.exp(u) + math.exp(-u))


def eq(x: list, y: list, m: int) -> bool:
    for i in range(0, m):
        if x[i] != y[i]:
            return False
    return True


def average(d_train: list, n: int, m: int) -> float:
    su = 0
    for i in range(0, n):
        su += d_train[i][m]
    return su / n


def solve(d_train: list, n: int, m: int, q: list, h_or_k: int, dist_type: Dist, kernel_type: Kernel,
          window_type: Window) -> float:
    # check if point exists
    su = 0.0
    cnt = 0
    for x in d_train:
        if eq(x, q, m):
            su += x[m]
            cnt += 1
    if cnt > 0:
        return su / cnt

    # calc_nadaraya_watson
    d_train = sorted(d_train, key=lambda x: calc_distance(dist_type, x, q, m))

    h = float(h_or_k) if window_type == Window.FIXED else calc_distance(dist_type, d_train[h_or_k], q, m)
    if h == 0.0:
        return average(d_train, n, m)

    num = 0.0
    denum = 0.0

    for x in d_train:
        kern_res = calc_kernel(kernel_type, calc_distance(dist_type, x, q, m), h)
        num += x[m] * kern_res
        denum += kern_res
    return average(d_train, n, m) if denum == 0 else (0 if num == 0 else num / denum)


if __name__ == '__main__':
    n, m = map(int, input().split())
    d_train = []
    q = []
    for _ in range(n):
        d_train.append(list(map(int, input().split())))
    q = list(map(int, input().split()))
    dist_type = Dist(input())
    kernel_type = Kernel(input())
    window_type = Window(input())
    h_or_k = int(input())
    print("{0:0.20f}".format(solve(d_train, n, m, q, h_or_k, dist_type, kernel_type, window_type)))
