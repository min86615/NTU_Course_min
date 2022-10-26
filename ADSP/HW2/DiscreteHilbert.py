import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("k",
                    help="Please enter filter size k",
                    type=int)
args = parser.parse_args()
k = args.k
N = 2*k + 1
interval_size = 0.0001
x = np.arange(0, 1+interval_size, interval_size)
R = np.zeros(len(x), dtype="complex_")
Hd = [None for i in range(len(x))]
for i in range(len(x)):
    if 0 < x[i] < 0.5:
        Hd[i] = -1j
    elif x[i] > 0.5:
        Hd[i] = 1j
    else:
        Hd[i] = 0
F = np.arange(2*k+1)/N
selected_H = [None for i in range(len(F))]
for i in range(len(F)):
    if 0 < F[i] < 0.5:
        selected_H[i] = -1j
    elif F[i] > 0.5:
        selected_H[i] = 1j
    else:
        selected_H[i] = 0
r1 = np.fft.ifft(selected_H)
r1 = np.roll(r1,k)
r1_real = np.real(r1)
r1_imag = np.imag(r1)
plt.stem(np.arange(len(r1_real))-k,r1_real)
plt.title("Impulse Response(r real part)")
plt.show()
# plt.stem(r1_imag)
# plt.show()
for i in range(len(R)):
    for a in range(len(r1)):
        # print(r1[a]*np.exp(-1j*2*x[i]*np.pi*(a-k)))
        R[i] = R[i] + r1[a]*np.exp(-1j*2*x[i]*np.pi*(a-k))
R_imag = np.imag(R)
plt.scatter(F, np.imag(selected_H))
plt.plot(x, np.imag(R))
plt.plot(x, np.imag(Hd))
plt.title("Frequency Response(R imag part)")
plt.show()