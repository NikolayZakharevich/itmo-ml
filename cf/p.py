if __name__ == '__main__':
    k1, k2 = map(int, input().split())
    n = int(input())

    row_sum = [0] * k1
    col_sum = [0] * k2
    all_sum = 0

    real = {}
    for _ in range(n):
        x, y = map(int, input().split())
        real[(x - 1, y - 1)] = real.get((x - 1, y - 1), 0) + 1
        row_sum[x - 1] += 1
        col_sum[y - 1] += 1
        all_sum += 1

    print(real)
    res = 0
    for k, real_value in real.values():
        i, j = k
        expected = row_sum[i] * col_sum[j] / all_sum
        res += (real_value * expected) ** 2 / expected

    print(res)