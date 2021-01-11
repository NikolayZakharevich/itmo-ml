import numpy as np

nodes = []


class Node:
    value = None
    d = None
    prev_node_idx = None

    def __init__(self, prev_node_idx):
        self.prev_node_idx = prev_node_idx

    def shape(self):
        return self.value.shape

    def init_d(self):
        self.d = np.zeros(self.value.shape)


class Var(Node):
    def __init__(self, value):
        super().__init__(None)
        self.value = value

    def calc(self):
        pass


class Relu(Node):
    a_inverse = None

    def __init__(self, prev_node_idx, a_inverse):
        super().__init__(prev_node_idx)
        self.a_inverse = a_inverse

    def calc(self):
        self.value = np.vectorize(self.rect)(nodes[self.prev_node_idx].value)

    def push_d(self):
        f = lambda v_and_d: (1 / self.a_inverse if v_and_d[0] < 0 else 1) * v_and_d[1]
        prev = nodes[self.prev_node_idx]
        prev.d += np.apply_along_axis(f, 3, np.stack([prev.value, self.d], axis=3))

    def rect(self, elem):
        return elem / self.a_inverse if elem < 0 else elem


class Pool(Node):
    s = None

    def __init__(self, prev_node_idx, s):
        super().__init__(prev_node_idx)
        self.s = s

    def iterate(self):
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                for k in range(self.shape()[2]):
                    yield i, j, k

    def calc(self):
        prev = nodes[self.prev_node_idx]
        s = self.s
        p, r, c = prev.shape()
        self.value = np.zeros((p, (r - s) // s + 1, (c - s) // s + 1))
        for i, j, k in self.iterate():
            self.value[i][j][k] = np.max(prev.value[i, j * s: (j + 1) * s, k * s: (k + 1) * s])

    def push_d(self):
        s = self.s
        for i, j, k in self.iterate():
            values = nodes[self.prev_node_idx].value[i, j * s: (j + 1) * s, k * s: (k + 1) * s]
            for jj, kk in np.argwhere(values == np.amax(values)):
                nodes[self.prev_node_idx].d[i][jj][kk] += self.d[i][j][k]


class Bias(Node):
    b = None
    d_params = None

    def __init__(self, prev_node_idx, *b):
        super().__init__(prev_node_idx)
        self.b = list(b)

    def calc(self):
        self.value = nodes[self.prev_node_idx].value.copy()
        for i in range(self.shape()[0]):
            self.value[i] += self.b[i]

    def push_d(self):
        nodes[self.prev_node_idx].d += self.d
        self.d_params = np.zeros(len(self.b))
        for i in range(self.shape()[0]):
            self.d_params[i] += np.sum(self.d[i])


class Cnv(Node):
    a = None
    h, k, s, p = None, None, None, None
    d_params = None
    padded_prev_value = None

    def __init__(self, prev_node_idx, *args):
        super().__init__(prev_node_idx)
        self.h, self.k, self.s, self.p = args[:4]
        prev = nodes[prev_node_idx]
        self.a = np.array(args[4:]).reshape((self.h, prev.shape()[0], self.k, self.k))

    def iterate(self):
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                for k in range(self.shape()[2]):
                    for ii in range(self.a.shape[1]):
                        for jj in range(self.a.shape[2]):
                            for kk in range(self.a.shape[3]):
                                yield i, j, k, ii, jj, kk

    def calc(self):
        prev = nodes[self.prev_node_idx]
        self.padded_prev_value = self.pad_value(prev.value)
        self.value = np.zeros(self.calc_shape())
        for i, j, k, ii, jj, kk in self.iterate():
            self.value[i][j][k] += self.padded_prev_value[ii][j * self.s + jj][k * self.s + kk] * self.a[i][ii][jj][kk]

    def push_d(self):
        prev = nodes[self.prev_node_idx]
        self.d_params = np.zeros(self.a.shape)
        s, p = self.s, self.p

        cnv_diff = np.zeros((prev.shape()[0], prev.shape()[1] + 2 * p, prev.shape()[2] + 2 * p))
        for i, j, k, ii, jj, kk in self.iterate():
            cnv_diff[ii][j * s + jj][k * s + kk] += self.d[i][j][k] * self.a[i][ii][jj][kk]
            self.d_params[i][ii][jj][kk] += self.d[i][j][k] * self.padded_prev_value[ii][j * s + jj][k * s + kk]

        for k in range(cnv_diff.shape[0]):
            for i in range(p, cnv_diff[k].shape[0] - p):
                for j in range(p):
                    self.unpad1(cnv_diff[k], i, j)
                for j in range(cnv_diff[k].shape[1] - p, cnv_diff[k].shape[1]):
                    self.unpad2(cnv_diff[k], i, j)
            for i in range(cnv_diff[k].shape[0] - p, cnv_diff[k].shape[0]):
                for j in range(cnv_diff[k].shape[1]):
                    self.unpad3(cnv_diff[k], i, j)
            for i in range(self.p):
                for j in range(cnv_diff[k].shape[1]):
                    self.unpad4(cnv_diff[k], i, j)
        for i in range(prev.shape()[0]):
            for j in range(prev.shape()[1]):
                for k in range(prev.shape()[2]):
                    prev.d[i][j][k] += cnv_diff[i][j + self.p][k + self.p]

    def calc_shape(self):
        prev = nodes[self.prev_node_idx]
        r = (prev.shape()[1] + 2 * self.p - self.k) // self.s + 1
        c = (prev.shape()[2] + 2 * self.p - self.k) // self.s + 1
        return self.h, r, c

    def pad_value(self, value):
        p = self.p
        return np.pad(value, ((0, 0), (p, p), (p, p)), self.pad_mode())

    def pad_mode(self):
        return None


class Cnvm(Cnv):
    def pad_mode(self):
        return 'reflect'

    def unpad1(self, cnv_diff_k, i, j):
        cnv_diff_k[i][2 * self.p - j] += cnv_diff_k[i][j]

    def unpad2(self, cnv_diff_k, i, j):
        cnv_diff_k[i][2 * cnv_diff_k.shape[1] - 2 * self.p - j - 2] += cnv_diff_k[i][j]

    def unpad3(self, cnv_diff_k, i, j):
        cnv_diff_k[2 * (cnv_diff_k.shape[0] - self.p) - i - 2][j] += cnv_diff_k[i][j]

    def unpad4(self, cnv_diff_k, i, j):
        cnv_diff_k[2 * self.p - i][j] += cnv_diff_k[i][j]


class Cnve(Cnv):
    def pad_mode(self):
        return 'edge'

    def unpad1(self, cnv_diff_k, i, j):
        cnv_diff_k[i][self.p] += cnv_diff_k[i][j]

    def unpad2(self, cnv_diff_k, i, j):
        cnv_diff_k[i][cnv_diff_k.shape[1] - self.p - 1] += cnv_diff_k[i][j]

    def unpad3(self, cnv_diff_k, i, j):
        cnv_diff_k[cnv_diff_k.shape[0] - self.p - 1][j] += cnv_diff_k[i][j]

    def unpad4(self, cnv_diff_k, i, j):
        cnv_diff_k[self.p][j] += cnv_diff_k[i][j]


class Cnvc(Cnv):
    def pad_mode(self):
        return 'wrap'

    def unpad1(self, cnv_diff_k, i, j):
        cnv_diff_k[i][j + cnv_diff_k.shape[1] - 2 * self.p] += cnv_diff_k[i][j]

    def unpad2(self, cnv_diff_k, i, j):
        cnv_diff_k[i][j - cnv_diff_k.shape[1] + 2 * self.p] += cnv_diff_k[i][j]

    def unpad3(self, cnv_diff_k, i, j):
        cnv_diff_k[i - cnv_diff_k.shape[0] + 2 * self.p][j] += cnv_diff_k[i][j]

    def unpad4(self, cnv_diff_k, i, j):
        cnv_diff_k[i + cnv_diff_k.shape[0] - 2 * self.p][j] += cnv_diff_k[i][j]


def get_by_node_type(node_type, *node_arhs):
    if node_type == 'relu':
        return Relu(*node_arhs)
    if node_type == 'pool':
        return Pool(*node_arhs)
    if node_type == 'bias':
        return Bias(*node_arhs)
    if node_type == 'cnvm':
        return Cnvm(*node_arhs)
    if node_type == 'cnve':
        return Cnve(*node_arhs)
    if node_type == 'cnvc':
        return Cnvc(*node_arhs)


if __name__ == '__main__':
    row = list(map(int, input().split()))
    nodes.append(Var(np.array(row[2:]).reshape((row[1], row[0], row[0]))))

    l = int(input())
    for prev_id in range(l):
        row = input().split()
        nodes.append(get_by_node_type(row[0], prev_id, *list(map(int, row[1:]))))

    for node in nodes:
        node.calc()
        node.init_d()

    nodes[-1].d = np.array(list(map(float, input().split()))).reshape(nodes[-1].shape())
    for node in reversed(nodes[1:]):
        node.push_d()

    for plane in nodes[-1].value:
        for row in plane:
            print(*row)

    for plane in nodes[0].d:
        for row in plane:
            print(*row, end=" ")
    print()

    for node in nodes:
        if isinstance(node, Bias) or isinstance(node, Cnv):
            print(*node.d_params.flatten())