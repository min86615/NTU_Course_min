
import matplotlib.pyplot as plt
import numpy as np
import time

def recSTFT(x, t, f, B):
    len_t = len(t)
    len_f = len(f)
    dt = (t[-1] - t[0]) / (len_t - 1)
    df = (f[-1] - f[0]) / (len_f - 1)
    N = round(1 / (dt * df))
    Q = round(B / dt)
    m = f / df
    Xnm = np.zeros((len_t, len_f))
    mod_m = np.array(np.mod(m, N)).astype(int)
    for i in range(Q+1, len(x)-Q):
        x1 = np.zeros(401)
        x1[:2*Q+1] = x[i-Q-1:(i+Q)]
        X = np.fft.fft(x1)
        Xnm[i, :] = X[mod_m] * dt * np.exp(1j*2*np.pi*(Q-(i-1))*m/N)
    return Xnm.T

x = np.zeros(601)
t = np.arange(601) * 0.05
x[:201] = np.cos(2*np.pi*t[:201])
x[201:401] = np.cos(6*np.pi*t[201:401])
x[401:] = np.cos(4*np.pi*t[401:])
f = (np.arange(201) * 0.05) - 5
B = 1
start = time.time()
y = recSTFT(x, t, f, B)
print("use {:f}".format(time.time()-start))
y = np.abs(y)/np.max(np.abs(y))*255
plt.imshow(y, cmap="gray", origin="lower")
plt.xticks(np.arange(0, 601, 100), np.arange(0, 31, 5))
plt.yticks(np.arange(0, 201, 20), np.arange(-5, 6, 1))
plt.xlabel("Time(sec)")
plt.ylabel("Frequency(Hz)")
plt.show()