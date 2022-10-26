import numpy as np

N = 1100
M = 7
complexity = []
for L in range(1,1100//2+1):
    current_complexity = (N/L)*3*(L+M-1)*(np.log2(L+M-1)+1)
    complexity.append(current_complexity)
min_L = np.argmin(np.array(complexity)) + 1
print(complexity[min_L-1])
# arange_N = np.arange(1100//2)
# complexity = 3 * (N / arange_N)
print("test")