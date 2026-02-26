def solve():
    n, m = map(int, input().split())
    l = []
    w = []
    h = []

    for _ in range(m):
        a, b, c = map(int, input().split())
        l.append(a)
        w.append(b)
        h.append(c)

    volumes = []
    for j in range(m):
        volumes.append(l[j] * w[j] * h[j])

    fib = [0] * (n + 1)
    fib[1] = 1
    if n >= 2:
        fib[2] = 2
    for i in range(3, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]

    total_volume = sum(fib[i] ** 3 for i in range(1, n + 1))
    max_cube_side = fib[n]

    result = []
    for j in range(m):
        smallest_dim = min(l[j], w[j], h[j])
        if volumes[j] >= total_volume and smallest_dim >= max_cube_side:
            result.append('1')
        else:
            result.append('0')

    print(''.join(result))

t = int(input())
for _ in range(t):
    solve()
