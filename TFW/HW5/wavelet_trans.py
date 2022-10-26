import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt

x = cv2.imread("Doge.png", cv2.IMREAD_GRAYSCALE)/255
def wavedbc10(x):
     g = np.array([0.0033, -0.0126, -0.0062, 0.0776, -0.0322,
          -0.2423, 0.1384, 0.7243, 0.6038, 0.1601])
     h = np.array([0.1601, -0.6038, 0.7243, -0.1384, -0.2423,
          0.0322, 0.0776, 0.0062, -0.0126, -0.0033])
     g= np.expand_dims(g, axis=0)
     h = np.expand_dims(h, axis=0)
     x = np.pad(x, int((np.size(g)-1))//2, mode='maximum')
     xg = signal.convolve2d(g, x, mode='valid')
     V1L = xg.T[::2,:].T
     xh = signal.convolve2d(h, x, mode='valid')
     V1H = xh.T[::2,:].T

     V1Lg = signal.convolve2d(g.T, V1L, mode='valid')
     X1L = V1Lg[::2,:]
     V1Lh = signal.convolve2d(h.T, V1L, mode='valid')
     X1H1 = V1Lh[::2,:]

     V1Hg = signal.convolve2d(g.T, V1H, mode='valid')
     X1H2 = V1Hg[::2,:]
     V1Hh = signal.convolve2d(h.T, V1H, mode='valid')
     X1H3 = V1Hh[::2,:]
     return X1L, X1H1, X1H2, X1H3
[X1L, X1H1, X1H2, X1H3]= wavedbc10(x)
f, axs = plt.subplots(2,2,figsize=(10,10))
axs[0, 0].axis("off")
axs[0, 0].imshow(X1L, "gray")
axs[0, 0].set_title("X1L")
axs[0, 1].imshow(X1H1, "gray")
axs[0, 1].set_title("X1H1")
axs[1, 0].imshow(X1H2, "gray")
axs[1, 0].set_title("X1H2")
axs[1, 1].imshow(X1H3, "gray")
axs[1, 1].set_title("X1H3")
plt.show()
def upsampling(input):
     height, width = input.shape
     output = np.zeros((height * 2, width))
     output[::2, :] = input
     return output
def iwavedbc10(X1L, X1H1, X1H2, X1H3):
     h1 = np.array([-0.0033, -0.0126, 0.0062, 0.0776, 0.0322,
          -0.2423, -0.1384, 0.7243, -0.6038, 0.1601])
     g1 = np.array([0.1601, 0.6038, 0.7243, 0.1384, -0.2423,
          -0.0322, 0.0776, -0.0062, -0.0126, 0.0033])
     g1= np.expand_dims(g1, axis=0)
     h1 = np.expand_dims(h1, axis=0)
     
     X1L = upsampling(X1L)
     X1H1 = upsampling(X1H1)
     X1H2 = upsampling(X1H2)
     X1H3 = upsampling(X1H3)   
     x0 = signal.convolve2d(X1L, g1.T) + signal.convolve2d(X1H1, h1.T)
     x1 = signal.convolve2d(X1H2, g1.T) + signal.convolve2d(X1H3, h1.T)
     X0 = upsampling(x0.T).T
     X1 = upsampling(x1.T).T
     result = signal.convolve2d(X0, g1) + signal.convolve2d(X1, h1)
     filter_size = int(((np.size(h1)-1)//2)*2)
     return result[filter_size: -filter_size, filter_size: -filter_size]
result = iwavedbc10(X1L, X1H1, X1H2, X1H3)  
plt.imshow(result, cmap='gray')
plt.show()
print("test")