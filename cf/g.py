from collections import Counter


class Leaf():
    id = None
    predicted_class = None

    def __init__(self, id: int, predicted_class: int):
        self.id = id
        self.predicted_class = predicted_class

    def print(self):
        print("C %d" % self.predicted_class)


class Node():
    id = None
    left = None
    right = None

    feature_idx = None
    b = None
    depth = None

    def __init__(self, id: int, feature_idx: int, b: float, left, right):
        self.id = id
        self.feature_idx = feature_idx
        self.b = b
        self.left = left
        self.right = right

    def print(self):
        print("Q %d %f %d %d" % (self.feature_idx + 1, self.b, self.left.id, self.right.id))
        self.left.print()
        self.right.print()


n_classes = None
n_features = None
max_depth = None

X_y = []
last_v_idx = 0


def X(sample_idx):
    return X_y[sample_idx]


def y(sample_idx):
    return X_y[sample_idx][n_features]


def major_class(u_indexes):
    counter = Counter(list(map(lambda sample_idx: y(sample_idx), u_indexes)))
    return counter.most_common(1)[0][0]


def learn_d3(u_begin, u_end, depth):
    global last_v_idx
    last_v_idx += 1

    if depth > max_depth:
        return Leaf(last_v_idx, major_class(range(u_begin, u_end)))

    y_0 = y(u_begin)
    same_class = True
    for sample_idx in range(u_begin, u_end):
        if y_0 != y(sample_idx):
            same_class = False
            break
    if same_class:
        return Leaf(last_v_idx, y_0)

    u_mid, feature_idx = split(u_begin, u_end)
    if u_begin == u_mid or u_mid == u_end:
        return Leaf(last_v_idx, major_class(range(u_begin, u_end)))

    return Node(
        last_v_idx,
        feature_idx,
        (X(u_mid - 1)[feature_idx] + X(u_mid)[feature_idx]) / 2,
        learn_d3(u_begin, u_mid, depth + 1),
        learn_d3(u_mid, u_end, depth + 1)
    )


def calc_sum_delta(classes_cnt, sample_idx, inc=True):
    sample_y = y(sample_idx)
    old_cnt = classes_cnt[sample_y]
    if inc:
        classes_cnt[sample_y] += 1
        return 2 * old_cnt + 1
    else:
        classes_cnt[sample_y] -= 1
        return -2 * old_cnt + 1


def split(u_begin, u_end):
    best_score, best_mid_idx, best_feature_idx = 4e18, 0, 0
    for feature_idx in range(n_features):
        X_y[u_begin:u_end] = sorted(X_y[u_begin:u_end], key=lambda x: x[feature_idx])
        left_cnt = [0] * (n_classes + 1)
        left_sum = 0

        right_cnt = [0] * (n_classes + 1)
        right_sum = sum(map(lambda idx: calc_sum_delta(right_cnt, idx), range(u_begin, u_end)))

        for sample_idx in range(u_begin, u_end):
            if sample_idx != u_begin:
                l = sample_idx - u_begin
                r = u_end - sample_idx
                score = 1 / l - left_sum / l  + 1 / r - right_sum / r
                if score < best_score:
                    best_score, best_mid_idx, best_feature_idx = score, sample_idx, feature_idx
            left_sum += calc_sum_delta(left_cnt, sample_idx)
            right_sum += calc_sum_delta(right_cnt, sample_idx, False)
    X_y[u_begin:u_end] = sorted(X_y[u_begin:u_end], key=lambda x: x[best_feature_idx])
    return best_mid_idx, best_feature_idx


if __name__ == '__main__':
    n_features, n_classes, max_depth = map(int, input().split())
    n = int(input())

    for _ in range(n):
        X_y.append(list(map(int, input().split())))
    root = learn_d3(0, n, 1)

    print(last_v_idx)
    root.print()
