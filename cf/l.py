from math import sqrt

if __name__ == '__main__':
    n = int(input())

    x_all = []
    y_all = []
    for _ in range(n):
        x, y = map(int, input().split())
        x_all.append(x)
        y_all.append(y)

    x_avg = sum(x_all) / len(x_all)
    y_avg = sum(y_all) / len(y_all)

    num = 0
    x_denum = 0
    y_denum = 0
    for i in range(n):
        x = x_all[i]
        y = y_all[i]
        num += (x - x_avg) * (y - y_avg)
        x_denum += (x - x_avg) ** 2
        y_denum += (y - y_avg) ** 2

    print (0 if num == 0 else num / sqrt(x_denum * y_denum))
