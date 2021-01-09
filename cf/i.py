import math

nodes = []


def get_matrix(element, n, m):
    return [[element for _ in range(m)] for _ in range(n)]


def zeros(n, m):
    return get_matrix(0, n, m)


def ones(n, m):
    return get_matrix(1, n, m)


class Node:
    value = None
    d = None

    def rows_cnt(self):
        return len(self.value)

    def cols_cnt(self):
        return len(self.value[0])

    def init_d(self):
        if self.d is None:
            self.d = zeros(self.rows_cnt(), self.cols_cnt())


class Var(Node):
    r = None
    c = None

    def __init__(self, r, c):
        self.r = r
        self.c = c

    def calc(self):
        pass


class Tnh(Node):
    x = None

    def __init__(self, x):
        self.x = x - 1

    def calc(self):
        self.value = [list(map(math.tanh, row)) for row in nodes[self.x].value]

    def push_d(self):
        x = nodes[self.x]
        for i in range(x.rows_cnt()):
            for j in range(x.cols_cnt()):
                x.d[i][j] += (1 - self.value[i][j] ** 2) * self.d[i][j]


class Rlu(Node):
    a_inverse = None
    x = None

    def __init__(self, a_inverse, x):
        self.a_inverse = a_inverse
        self.x = x - 1

    def calc(self):
        self.value = [list(map(self.rect, row)) for row in nodes[self.x].value]

    def push_d(self):
        x = nodes[self.x]
        for i in range(x.rows_cnt()):
            for j in range(x.cols_cnt()):
                x.d[i][j] += (1 / self.a_inverse if x.value[i][j] < 0 else 1) * self.d[i][j]

    def rect(self, elem):
        return elem / self.a_inverse if elem < 0 else elem


class Mul(Node):
    a, b = None, None

    def __init__(self, a, b):
        self.a = a - 1
        self.b = b - 1

    def calc(self):
        a, b = nodes[self.a], nodes[self.b]
        self.value = zeros(a.rows_cnt(), b.cols_cnt())
        for i in range(a.rows_cnt()):
            for j in range(b.cols_cnt()):
                for k in range(a.cols_cnt()):
                    self.value[i][j] += a.value[i][k] * b.value[k][j]

    def push_d(self):
        a, b = nodes[self.a], nodes[self.b]
        for i in range(a.rows_cnt()):
            for j in range(a.cols_cnt()):
                for k in range(b.cols_cnt()):
                    a.d[i][j] += self.d[i][k] * b.value[j][k]
        for j in range(a.cols_cnt()):
            for k in range(b.cols_cnt()):
                for i in range(a.rows_cnt()):
                    b.d[j][k] += a.value[i][j] * self.d[i][k]


class Sum(Node):
    args = []

    def __init__(self, _, *args):
        self.args = list(map(lambda x: x - 1, list(args)))

    def calc(self):
        n, m = nodes[self.args[0]].rows_cnt(), nodes[self.args[0]].cols_cnt()
        self.value = zeros(n, m)
        for i in range(n):
            for j in range(m):
                for arg in self.args:
                    self.value[i][j] += nodes[arg].value[i][j]

    def push_d(self):
        n, m = nodes[self.args[0]].rows_cnt(), nodes[self.args[0]].cols_cnt()
        for i in range(n):
            for j in range(m):
                for arg in self.args:
                    nodes[arg].d[i][j] += self.d[i][j]


class Had(Node):
    args = []

    def __init__(self, _, *args):
        self.args = list(map(lambda x: x - 1, list(args)))

    def calc(self):
        n, m = self.get_n_and_m()
        self.value = ones(n, m)
        for i in range(n):
            for j in range(m):
                for arg in self.args:
                    self.value[i][j] *= nodes[arg].value[i][j]

    def push_d(self):
        n, m = self.get_n_and_m()
        for i in range(n):
            for j in range(m):
                for x in range(len(self.args)):
                    acc = self.d[i][j]
                    for y in range(len(self.args)):
                        if x != y:
                            acc *= nodes[self.args[y]].value[i][j]
                    nodes[self.args[x]].d[i][j] += acc

    def get_n_and_m(self):
        return nodes[self.args[0]].rows_cnt(), nodes[self.args[0]].cols_cnt()

def get_by_node_type(node_type, *node_arhs):
    if node_type == 'var':
        return Var(*node_arhs)
    if node_type == 'tnh':
        return Tnh(*node_arhs)
    if node_type == 'rlu':
        return Rlu(*node_arhs)
    if node_type == 'mul':
        return Mul(*node_arhs)
    if node_type == 'sum':
        return Sum(*node_arhs)
    if node_type == 'had':
        return Had(*node_arhs)


if __name__ == '__main__':
    n, m, k = list(map(int, input().split()))
    for _ in range(n):
        row = input().split()
        nodes.append(get_by_node_type(row[0], *list(map(int, row[1:]))))
    for node_idx in range(m):
        value = []
        for i in range(nodes[node_idx].r):
            value.append(list(map(int, input().split())))
        nodes[node_idx].value = value

    for node in nodes:
        node.calc()
        node.init_d()

    for node_idx in range(n - k, n):
        nodes[node_idx].d = []
        for _ in range(nodes[node_idx].rows_cnt()):
            nodes[node_idx].d.append(list(map(float, input().split())))

    for node in reversed(nodes[m:]):
        node.push_d()

    for node in nodes[n - k:]:
        for row in node.value:
            print(*row)

    for node in nodes[:m]:
        for row in node.d:
            print(*row)
