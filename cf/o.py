if __name__ == '__main__':
    k = int(input())
    n = int(input())

    # { x1: {y1: cnt_y11, y2: cnt_y12}, x2: {y1: cnt_y21, y2: cnt_y22} },
    y_cnt_map = {}
    x_cnt = {}
    for _ in range(n):
        x, y = map(int, input().split())
        if x not in y_cnt_map:
            y_cnt_map[x] = {y: 1}
        else:
            y_cnt_map[x][y] = y_cnt_map[x].get(y, 0) + 1
        x_cnt[x] = x_cnt.get(x, 0) + 1

    EY = []
    for x, y_cnt_per_x in y_cnt_map.items():
        cnt_y_all, numerator, numerator2 = 0, 0, 0
        for y, cnt_y in y_cnt_per_x.items():
            cnt_y_all += cnt_y
            numerator += y * cnt_y
            numerator2 += y * y * cnt_y
        EY.append((x, numerator / cnt_y_all, numerator2 / cnt_y_all))

    res = 0
    for x, e, e2 in EY:
       res += x_cnt[x] * (e2 - e * e)
    print (res / n)
