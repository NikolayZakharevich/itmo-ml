if __name__ == '__main__':
    n = int(input())

    a = []
    c_row = [0] * n
    c_col = [0] * n

    all = 0

    for _ in range(n):
        a.append(list(map(int, input().split())))

    for i in range(n):
        for j in range(n):
            c_row[i] += a[i][j]
            c_col[j] += a[i][j]
        all += c_row[i]

    f_micro = 0.0
    for i in range(n):
        precision_c = a[i][i] / c_row[i] if a[i][i] > 0.0 else 0.0
        recall_c = a[i][i] / c_col[i] if a[i][i] > 0.0 else 0.0
        numerator = precision_c * recall_c

        f_micro += c_row[i] * 2 * (precision_c * recall_c) / (precision_c + recall_c) if numerator > 0.0 else 0.0

    f_micro /= all

    prec_w = 0.0
    for i in range(n):
        num = a[i][i] * c_row[i]
        prec_w += num / c_col[i] if num > 0 else 0

    prec_w /= all
    recall_w = 0.0
    for i in range(n):
        recall_w += a[i][i]
    recall_w /= all
    numerator = prec_w * recall_w
    f_macro = 2 * prec_w * recall_w / (prec_w + recall_w) if numerator > 0 else 0.0

    print("{0:0.20f}\n{1:0.20f}".format(f_macro, f_micro))
