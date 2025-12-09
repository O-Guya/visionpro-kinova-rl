
def climb(n):
    a = []
    for i in range(0,n):
        a.append(0)
    a[0] = 1
    a[1] = 2
    for i in range(2,n):
        a[i] = a[i-1] + a[i-2]
    return a[n-1]

print(climb(100))
