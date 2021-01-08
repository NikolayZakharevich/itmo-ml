if __name__ == '__main__':
    m = int(input())
    n = 2 ** m
    f_all = []
    for _ in range(n):
        f_all.append(int(input()))
    print(2)
    print(n, 1)
    for args in range(n):
        for bit in range(m):
            print(1.0 if args & (1 << bit) else -1e9, end=" ")
        print(0.5 - bin(args).count('1'))
    print(*f_all)
    print(-0.5)
