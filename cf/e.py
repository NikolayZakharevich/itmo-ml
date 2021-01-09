import random
import time

EPS = 1e-10

K = []
y = []
alpha = []
b = 0
n = 0
c = 0


def f(j):
    return sum(map(lambda i: alpha[i] * y[i] * K[i][j], range(n))) + b


def calc_l_and_h(i, j):
    if y[i] != y[j]:
        return max(0., alpha[j] - alpha[i]), min(c, c + alpha[j] - alpha[i])
    return max(0., alpha[i] + alpha[j] - c), min(c, alpha[i] + alpha[j])


def calc_nu(i, j):
    return 2 * K[i][j] - K[i][i] - K[j][j]


if __name__ == '__main__':
    n = int(input())
    for _ in range(n):
        row = list(map(int, input().split()))
        K.append(row[:-1])
        y.append(row[-1])
    c = int(input())

    alpha = [0.] * n
    start_time = time.process_time()
    while time.process_time() - start_time < 2:
        for i in range(n):
            e_i = f(i) - y[i]
            if (y[i] * e_i < -EPS and alpha[i] < c) or (y[i] * e_i > EPS and alpha[i] > 0):
                j = i
                while j == i:
                    j = random.choice(range(n))
                e_j = f(j) - y[j]
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                l, h = calc_l_and_h(i, j)
                if l == h:
                    continue
                nu = calc_nu(i, j)
                if nu >= 0:
                    continue
                alpha[j] -= y[j] * (e_i - e_j) / nu
                if alpha[j] > h:
                    alpha[j] = h
                elif alpha[j] < l:
                    alpha[j] = l
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                calc_b1 = lambda: b - e_i - y[i] * (alpha[i] - alpha_i_old) * K[i][i] - y[j] * (
                        alpha[j] - alpha_j_old) * K[i][j]
                calc_b2 = lambda: b - e_j - y[i] * (alpha[i] - alpha_i_old) * K[i][j] - y[j] * (
                        alpha[j] - alpha_j_old) * K[j][j]

                if 0 < alpha[i] < c:
                    b = calc_b1()
                elif 0 < alpha[j] < c:
                    b = calc_b2()
                else:
                    b = (calc_b1() + calc_b2()) / 2
    for a in alpha:
        print("%.20f" % a)
    print(b)