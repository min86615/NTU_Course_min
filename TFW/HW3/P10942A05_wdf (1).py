
import matplotlib.pyplot as plt
import numpy as np
import time

def wdf(x, t, f):
    dt = t[1] - t[0]
    df = f[1] - f[0]
    N = np.round(1/(2*dt*df))
    n0 = np.round(t[0]/dt)
    n1 = np.round(t[-1]/dt)
    m0 = np.round(f[0]/df)
    m1 = np.round(f[-1]/df)
    mod_m = np.array(np.mod(np.arange(m0 , m1+1), N)).astype(int)
    y = np.zeros((int(m1-m0+1), int(n1-n0+1)))
    
    for n in np.arange(n0, n1+1):
        Q = min(n1-n, n-n0)
        w = x[(np.arange(n-Q, n+Q+1)-n0).astype('int')]*np.conj(x[(np.arange(n+Q, n-Q-1, -1)-n0).astype('int')]).T
        W = np.fft.fft(w, int(N))*2*dt
        y[:, int(n-n0)] = W[mod_m]*np.exp(1j*2*np.pi/N*Q*(mod_m)).T
    return y
        

dt = 0.0125
t = np.arange(-9, 9.0125, dt)
df = 0.025
f = np.arange(-4, 4.025, df)
x = np.exp(1j*t**2/10-1j*3*t)*((t>=-9)&(t<=1))+np.exp(1j*t**2/2+1j*6*t)*np.exp(-(t-4)**2/10)
start = time.time()
y = wdf(x, t, f)
time_usage = time.time()-start
y = np.abs(y)/np.max(np.abs(y))*400
print(y.shape)
plt.imshow(y, cmap="gray", origin="lower", aspect='auto')
plt.xticks(np.arange(0, 1500, 200), np.arange(-9, 10, 2.5))
plt.yticks(np.arange(0, 321, 50), np.arange(0, 321, 50)/40 -4)
plt.title('WDF')
plt.title('Use: {:.2f}(s)'.format(time_usage), loc='right')
plt.xlabel("Time(sec)")
plt.ylabel("Frequency(Hz)")
plt.show()