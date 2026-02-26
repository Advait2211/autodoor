def solve():

    x = int(input())
    day_before_yesterday = 0
    yesterday = 0
    today = 0
    cycles = 0

    # if x == 1:
    #     return 3
    # if x == 2:
    #     return 5
    # if x == 3:
    #     return 6
    # if x == 4:
    #     return 7
    # if x == 5:
    #     return 7

    while today < x:
        temp = today
        today = yesterday * 2 + 1
        day_before_yesterday = yesterday
        yesterday = temp
        cycles += 1
        # print(day_before_yesterday, yesterday, today, cycles)

    return cycles + 2

t = int(input())

for _ in range(t):
    print(solve())