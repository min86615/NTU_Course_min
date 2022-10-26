from numpy.core.fromnumeric import shape
from numpy.fft import fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LogNorm 
img = cv2.imread("481544_Photographer_degraded-HW4.png", cv2.IMREAD_GRAYSCALE)



def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result/result.sum()

def WienerFilter(input, kernel, K):
    fft_filter = fft2(kernel, input.shape)
    w = np.conj(fft_filter) / (np.abs(fft_filter)**2 + K)
    result = ifft2(w * fft2(input))
    return np.clip(np.abs(result), 0, 255)

filter_size = 19
horizontal_line = np.zeros((filter_size, filter_size))
horizontal_line[:, int((filter_size+1)//2):int((filter_size+1)//2+1)] = 1
rotate_kernel = rotate_image(horizontal_line, -45)
plt.imshow(np.abs(rotate_kernel), 'gray')
plt.show()
result = WienerFilter(img, rotate_kernel, 0.08)
plt.imshow(result, 'gray')
plt.show()