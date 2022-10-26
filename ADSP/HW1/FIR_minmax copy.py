from cmath import pi
from re import L, S
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
# passband 0-1200
# sampling freq 6000Hz
# transition band 1200-1500
# Weighted 1 for passband 0.6 for stopband
delta = 0.001
N = 9
k = int((N-1)/2)

# follow slider use k+2 size
matrix_size = k+2
# AS = H
A = np.zeros((matrix_size, matrix_size))
S = np.zeros((matrix_size))

# set F points (k+2) not in transition band (0.2-0.25)
# 10 point
F = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
f = np.arange(0, 0.5+delta, delta)

Hd = (f <= 0.225)*1
weight = np.zeros(len(f))
passweight = (f <= 0.2)*1
rejectweight = (f >= 0.25)*0.6
weight = passweight + rejectweight
min_error = 2222

def weight_function(F):
    if F>= 0.25:
        return 0.6
    elif F <= 0.2:
        return 1
    else:
        return 0 


E0 = 4 # current err
E1 = 64 # previous iter err
iter_times = 0
#check whether next iter
while (((E1-E0)>delta) or (E1-E0<0)):
# while min_error>0.1:
    H = (np.array(F) < 0.2)*1
    # step2
    for i in range(matrix_size):
        for j in range(matrix_size-1):
            A[i,j] = np.cos(2*np.pi*j*F[i])
        A[i,-1] = ((-1)**(i))/weight_function(F[i])
    S = (np.linalg.inv(A))@H
    # step 3
    RF = 0
    # due to last value of S is e
    for i in range(len(S)-1):
        RF += S[i]*np.cos(2*np.pi*i*f)

    error = (RF-Hd)*weight
    # plt.plot(RF)
    # plt.plot(error)
    # 
    # pass head and tail
    new_F_without_head_tail = []
    new_F_val_without_head_tail = []
    max_min_loc = 1
    for i in range(2,len(error)-1):
        # local max
        if (error[i] > error[i-1]) and (error[i] > error[i+1]):
            # F[max_min_loc] = delta*i
            new_F_without_head_tail.append(delta*i)
            new_F_val_without_head_tail.append(error[i])
            max_min_loc += 1
            # plt.scatter(i, error[i], c='b')
        # local min
        if (error[i] < error[i-1]) and (error[i] < error[i+1]):
            # F[max_min_loc] = delta*i
            new_F_without_head_tail.append(delta*i)
            new_F_val_without_head_tail.append(error[i])
            max_min_loc += 1
            # plt.scatter(i, error[i], c='r')
        # if max_min_loc == 9:
        #     break
    new_F_val_without_head_tail.append(error[0])
    new_F_without_head_tail.append(0)
    new_F_val_without_head_tail.append(error[-1])
    new_F_without_head_tail.append(0.5)
    new_F_select = np.argsort(np.abs(new_F_val_without_head_tail))[::-1][:10]
    # step 5
    E1 = E0
    iter_times += 1
    E0 = max(abs(error))
    # print(E0)
    if min_error > E0:
        min_error = E0
        print("minerror: %s" % min_error)
    F = []
    for F_loc in new_F_select:
       F.append(new_F_without_head_tail[F_loc])
    while len(F) < matrix_size:
        rand_num = 0.5*np.random.rand(1)
        while 0.25>rand_num>0.2:
            rand_num = 0.5*np.random.rand(1)
        F.append(rand_num)
        # if np.mean(F) >= 0.25:
        #     F.append(0.5)
        # else:
        #     F.append(0)
    F = np.sort(F)
    # plt.show()
plt.plot(RF)
plt.plot(error)
h = np.zeros(N)
plt.show()
h[k] = S[0]
for i in range(k):
    h[k-1-i] = S[i+1]/2
    h[k+1+i] = S[i+1]/2
plt.scatter(np.arange(N), h)
plt.show()
print("break!")