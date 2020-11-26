if __name__ == '__main__':
    n = int(input())

    pairs = []

    x_all = []
    y_all = []
    x_rank = [0] * n
    y_rank = [0] * n
    for i in range(n):
        x, y = map(int, input().split())
        x_all.append((x, i))
        y_all.append((y, i))

    x_all.sort()
    y_all.sort()
    for i in range(n):
        x, pos_x = x_all[i]
        y, pos_y = y_all[i]
        x_rank[pos_x] = i
        y_rank[pos_y] = i


    sum = 0
    for i in range(n):
        sum += (x_rank[i] - y_rank[i]) ** 2
    print(1 - (6 * sum) / (n * (n * n - 1)))
