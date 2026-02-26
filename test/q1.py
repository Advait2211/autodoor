"""
7
1
5
14
2025
31415
536870910
1000000000
"""

def solve():
    x = int(input())
    day_before_yesterday = 1
    yesterday = 2
    today = 3
    cycles = 4

    if x == 1:
        return 3
    if x == 2:
        return 5
    if x == 3:
        return 6

    while today < x:
        temp = today
        today = day_before_yesterday * 2 + 1
        day_before_yesterday = yesterday
        yesterday = temp
        cycles += 1

    return cycles



    # while True:
    #     min_index = arr.index(min(arr))
    #     arr[min_index] = arr[min_index] * 2 + 1
    #     cycles += 1
    #     if arr[min_index] >= x:
    #         return cycles
        # old_val = 0
        # new_val = 2
        # temp = 3
        # cycles = 4

        # while new_val < x:
        #     new_val = old_val * 2 + 1
        #     temp = old_temp
        #     cycles += 1
        # return cycles



t = int(input())

for _ in range(t):
    print(solve())