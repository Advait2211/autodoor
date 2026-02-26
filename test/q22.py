import sys

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        arr = list(map(int, input[ptr:ptr + n]))
        ptr += n
        
        if all(x == arr[0] for x in arr):
            print(0)
            continue
        
        # Compute prefix sums and prefix costs
        prefix = [0] * n
        prefix[0] = 0
        for i in range(1, n):
            prefix[i] = prefix[i - 1] + arr[i] * i
        
        # Compute suffix sums and suffix costs
        suffix = [0] * n
        suffix[-1] = 0
        for i in range(n - 2, -1, -1):
            suffix[i] = suffix[i + 1] + arr[i] * (n - 1 - i)
        
        # Find the minimal total cost
        min_cost = float('inf')
        for i in range(n):
            total = prefix[i] + suffix[i]
            if total < min_cost:
                min_cost = total
        print(min_cost)

solve()