from numpy.core.fromnumeric import shape
from numpy.fft import fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LogNorm 
img = cv2.imread("481544_Photographer_degraded-HW4.png", cv2.IMREAD_GRAYSCALE)



def motion_kernel(img, angle, filter_size):
    kernel = np.zeros(img.shape)
    center_x, center_y = img.shape[0]/2, (img.shape[1])/2
    slope = np.tan(angle*np.pi/180)
    for i in np.arange(-filter_size, filter_size, 1):
        kernel[int(center_x+i), int(np.round(center_y+i*slope))] = 1
    return kernel/kernel.sum()
        
def WienerFilter(input, kernel, K):
    fft_filter = fft2(kernel, input.shape)
    
    plt.imshow(np.abs(fft_filter), 'gray')
    plt.show()
    w = np.conj(fft_filter) / (np.abs(fft_filter)**2 + K)
    result = ifft2(w * fft2(input))
    plt.imshow(np.abs(w), 'gray')
    plt.show()
    return np.clip(np.abs(np.fft.fftshift(result)), 0, 255)


fft_img = np.fft.fft2(img)
spec_img = np.fft.fftshift(fft_img)
plt.imshow(np.abs(spec_img), norm=LogNorm(vmin=5))
plt.show()
kernel = motion_kernel(img, -45, 6)
plt.imshow(np.abs(kernel), 'gray')
plt.show()
result = WienerFilter(img, kernel, 0.25)
plt.imshow(result, 'gray')
plt.show()
fft_img = np.fft.fft2(result)
spec_img = np.fft.fftshift(fft_img)
plt.imshow(np.abs(spec_img), norm=LogNorm(vmin=5))
plt.show()