
import matplotlib.pyplot as plt
import numpy as np
import time

def hht(x, t, f):
    local_min = []
    local_max = []
    tmp_x_min = 10
    tmp_x_max = -10
    for idx_x in range(len(x)):
                                                  
    return y
        

dt = 0.01
t = np.arange(0, 10.01, dt)
x = 0.2*t + np.cos(2*np.pi*t) + 0.4*np.cos(10*np.pi*t)
thr = 0.2
y = hht(x, t, thr)
plt.plot(t, x)
plt.show()
# df = 0.025
# f = np.arange(-4, 4.025, df)
# x = np.exp(1j*t**2/10-1j*3*t)*((t>=-9)&(t<=1))+np.exp(1j*t**2/2+1j*6*t)*np.exp(-(t-4)**2/10)
# start = time.time()
# y = wdf(x, t, f)
# time_usage = time.time()-start
# y = np.abs(y)/np.max(np.abs(y))*400
# print(y.shape)
# plt.imshow(y, cmap="gray", origin="lower", aspect='auto')
# plt.xticks(np.arange(0, 1500, 200), np.arange(-9, 10, 2.5))
# plt.yticks(np.arange(0, 321, 50), np.arange(0, 321, 50)/40 -4)
# plt.title('WDF')
# plt.title('Use: {:.2f}(s)'.format(time_usage), loc='right')
# plt.xlabel("Time(sec)")
# plt.ylabel("Frequency(Hz)")
# plt.show()