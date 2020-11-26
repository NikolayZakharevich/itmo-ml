def calc_row(row):
    sum = 0
    n = len(row)
    r = 0
    for i in range(n - 1):
        sum += row[i]
        r += (2 * n - (2 * i + 2)) * row[i]
    sum += row[n - 1]
    return (n - 1) * sum - r


if __name__ == '__main__':
    _ = int(input())
    n = int(input())
    per_y_map = {}

    all = []
    for i in range(n):
        x, y = map(int, input().split())
        if y not in per_y_map:
            per_y_map[y] = [x]
        else:
            per_y_map[y].append(x)
        all.append(x)

    all.sort()
    inner = 0
    for y, x_per_y in per_y_map.items():
        x_per_y.sort()
        inner += calc_row(x_per_y)

    print(inner * 2)
    print((calc_row(all) - inner) * 2)