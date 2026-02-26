def find_consecutive_same_elements(arr):
    n = len(arr)
    result = []
    start = 0

    while start < n:
        end = start
        while end + 1 < n and arr[end + 1] == arr[start]:
            end += 1
        if end > start:
            result.append((start, end))
        start = end + 1

    return result

def solve():
    n = int(input())
    arr = list(map(int, input().split()))

    consecutive_indices = find_consecutive_same_elements(arr)
    # print(consecutive_indices)

    minimum = min(arr)
    maximum = max(arr)

    if minimum == maximum:
        return 0

    min_index = arr.index(minimum)

    if not consecutive_indices:
        left = min(arr) * (min_index - 1)
        right = min(arr) * (n - min_index)
        return left + right
    else:
        left = min(arr) * (min_index - 1)
        right = min(arr) * (n - min_index)
        temp = left + right
        for val in consecutive_indices:
            left = val[0] * arr[val[0]]
            right = (n - (val[1] + 1)) * arr[val[1]]
            # print(left, right)
            if left + right < temp:
                temp = left + right

        return temp

t = int(input())
for _ in range(t):
    print(solve())