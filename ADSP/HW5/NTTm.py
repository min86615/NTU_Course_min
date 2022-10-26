import numpy as np

def NTTm(M, N):
    a = 2
    a_inv = 1
    N_inv = 1
    while (((a**N % M)!=1) | (np.sum((a**np.arange(1, N) % M) == 1) != 0)) & (a <= M-1):
        a += 1
    if a == M:
        print("Can't find primitive root")
        return None
    fwd_NTT = np.zeros((N, N))
    inv_NTT = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cell_val = 1
            for _ in range(int(i*j)):
                cell_val = (cell_val * a % M)
            fwd_NTT[i, j] = cell_val
        

    while ((N*N_inv) % M) != 1:
        N_inv += 1

    while ((a*a_inv) % M) != 1:
        a_inv += 1

    for i in range(N):
        for j in range(N):
            inv_cell_val = 1
            for _ in range(i * j):
                inv_cell_val = (inv_cell_val * a_inv) % M
            # inv_NTT[i, j] = ((a_inv**(i * j))*N_inv)% M
            inv_NTT[i, j] = (inv_cell_val * N_inv) % M
    return fwd_NTT, inv_NTT
M = 5
N = 4
fwd_NTT, inv_NTT = NTTm(M, N)
print("Forward NTTs")
print(fwd_NTT)
print("Inverse NTTs")
print(inv_NTT)
