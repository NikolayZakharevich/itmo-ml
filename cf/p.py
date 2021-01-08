if __name__ == '__main__':
    k1, k2 = map(int, input().split())
    n = int(input())

    row_sum = [0] * k1
    col_sum = [0] * k2
    all_sum = 0

    real = {}
    for _ in range(n):
        x, y = map(int, input().split())
        x -= 1
        y -= 1
        real[(x, y)] = real.get((x, y), 0) + 1
        row_sum[x] += 1
        col_sum[y] += 1

    res = n
    for k in real:
        i, j = k
        expected = row_sum[i] * col_sum[j] / n
        res += (real[k] - expected) ** 2 / expected - expected
    print(res)
