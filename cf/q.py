from math import log

if __name__ == '__main__':
    kx, ky = map(int, input().split())
    n = int(input())
    x_pr, xy_pr = {}, {}
    for _ in range(n):
        x, y = tuple(map(int, input().split()))
        prob = 1 / n
        x_pr[x] = x_pr.get(x, 0) + prob
        xy_pr[(x, y)] = xy_pr.get((x, y), 0) + prob
    print(sum(map(lambda x_y: -xy_pr[(x_y[0], x_y[1])] * (log(xy_pr[(x_y[0], x_y[1])]) - log(x_pr[x_y[0]])), xy_pr)))
