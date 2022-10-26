
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import wavfile

def Gabor(x, tau, t, f, sgm):
    C = len(t)
    F = len(f)
    T = len(tau)
    dtau = tau[1] - tau[0]
    dt = t[1] - t[0]
    df = f[1] - f[0]
    n0 = np.round(tau/dtau)
    m0 = np.round(f/df)
    c0 = np.round(t/dt)
    N = round(1 / (dtau * df))
    Q = round(1.9143 / (dtau*np.sqrt(sgm)))
    S = round(dt/dtau)
    Xfc = np.zeros((F, C))
    mod_m = np.array(np.mod(m0, N)).astype(int)
    for i in range(C):
        x1 = np.zeros(N-2*Q-1)
        q = np.arange(2*Q+1)
        windowfn = np.exp(-sgm*np.pi*((Q-q)*dtau)**2)
        selectitem = x[np.clip(q-Q+(i*S), 0, T-1)]
        x1[:2*Q+1] = windowfn*selectitem
        X = np.fft.fft(x1, N)
        Xfc[:, i] = X[mod_m] * dtau * np.exp(1j*2*np.pi*(Q-S*(i-1))*m0/N)
    return Xfc

fs_rate, wave_data = wavfile.read("Chord.wav")
wave_data = wave_data / np.max(abs(wave_data))
tau = np.arange(0, wave_data.shape[0])*1/fs_rate
dt = 0.01
t = np.arange(0, max(tau), dt)
df = 1
f = np.arange(20, 1000, df)
sgm = 200
x = wave_data[:, 0]
start = time.time()
y = Gabor(x, tau, t, f, sgm)
time_usage = time.time()-start
y = np.abs(y)/np.max(np.abs(y))*255
y_ticks = np.arange(0, 1000, 200)
y_ticks[0] += 20
plt.imshow(y, cmap="gray", origin="lower", aspect='auto')
plt.xticks(np.arange(0, 160, 100), np.arange(0, 1.6, 1))
plt.yticks(np.arange(0, 980, 200), y_ticks)
plt.title('Gabor Transform')
plt.title('Use: {:.2f}(s)'.format(time_usage), loc='right')
plt.xlabel("Time(sec)")
plt.ylabel("Frequency(Hz)")
plt.show()