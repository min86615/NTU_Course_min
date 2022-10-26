import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 

keyboard_img = cv2.imread("480548_keyboard.tif", cv2.IMREAD_GRAYSCALE)
fft_img = np.fft.fft2(keyboard_img)
spec_img = np.fft.fftshift(fft_img)
plt.imshow(np.abs(spec_img), norm=LogNorm(vmin=5))
plt.show()

kernel=np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype="int")
# odd symmetry the kernel change to below I don't know the question "exactly" odd symmetry means
# the original also odd symmetry in y axis 
# kernel=np.array([[0, -1, -2],[1, 0, -1],[2, 1, 0]], dtype="int")
padding_size = (keyboard_img.shape[0] - kernel.shape[0], keyboard_img.shape[1] - kernel.shape[1])
kernel = np.pad(kernel, (((padding_size[0]+1)//2, padding_size[0]//2), ((padding_size[1]+1)//2, padding_size[1]//2)), 'constant')
kernel = np.fft.ifftshift(kernel)
freq_sobel_img = np.real(np.fft.ifft2(np.fft.fft2(keyboard_img) * np.fft.fft2(kernel)))
plt.imshow(freq_sobel_img, 'gray')
plt.show()

def imgfilter2D(img, filter, ratio):
    filter = np.flip(filter)
    img_x, img_y = img.shape
    filter_x, filter_y = filter.shape
    result_mtx = np.zeros(((img_x - filter_x + 1), (img_y - filter_y + 1)))
    for i in range(result_mtx.shape[0]):
        for j in range(result_mtx.shape[1]):
            result_mtx[i][j] = np.sum(img[i:i + filter_x, j:j + filter_y] * filter) / ratio
    result_mtx = ((result_mtx/np.max(np.abs(result_mtx))) + 1)*127.5
    return result_mtx.astype('uint8')

kernel=np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype="int")
result_sobel_y = imgfilter2D(keyboard_img, kernel, 4)
plt.imshow(result_sobel_y, 'gray')
plt.show()

kernel=np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype="int")
kernel = np.rot90(kernel)
padding_size = (keyboard_img.shape[0] - kernel.shape[0], keyboard_img.shape[1] - kernel.shape[1])
kernel = np.pad(kernel, (((padding_size[0]+1)//2, padding_size[0]//2), ((padding_size[1]+1)//2, padding_size[1]//2)), 'constant')
kernel = np.fft.ifftshift(kernel)
freq_sobel_img = np.real(np.fft.ifft2(np.fft.fft2(keyboard_img) * np.fft.fft2(kernel)))
plt.imshow(freq_sobel_img, 'gray')
plt.show()