import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

img_newspaper = cv2.imread("480548_newspaper.tif", cv2.IMREAD_GRAYSCALE)
fft_img = np.fft.fft2(img_newspaper)
spec_img = np.fft.fftshift(fft_img)
plt.imshow(np.abs(spec_img), norm=LogNorm(vmin=5))
plt.show()
reject_filter = np.ones(img_newspaper.shape)
reject_filter[:, 120:130] = 0
reject_filter[:, 175:185] = 0
reject_filter[190:255, 120:130] = 1
reject_filter[190:255, 175:185] = 1
reject_filter[170:190, :] = 0
reject_filter[255:275, :] = 0
reject_filter[170:190, 130:175] = 1
reject_filter[255:275, 130:175] = 1
plt.imshow(reject_filter*255, "gray")
plt.show()
filter_spectrum = spec_img * reject_filter
filter_spectrum = np.fft.ifftshift(filter_spectrum)
result_img = np.fft.ifft2(filter_spectrum)
plt.imshow(np.abs(result_img), "gray")
plt.show()

img_cassini = cv2.imread("480548_cassini.tif", cv2.IMREAD_GRAYSCALE)
fft_img = np.fft.fft2(img_cassini)
spec_img = np.fft.fftshift(fft_img)
plt.imshow(np.abs(spec_img), norm=LogNorm(vmin=5))
plt.show()
reject_filter = np.ones(img_cassini.shape)
reject_filter[:, 335:339] = 0
reject_filter[330:350, 335:339] = 1
plt.imshow(reject_filter*255, "gray")
plt.show()
filter_spectrum = spec_img * reject_filter
filter_spectrum = np.fft.ifftshift(filter_spectrum)
result_img = np.fft.ifft2(filter_spectrum)
plt.imshow(np.abs(result_img), "gray")
plt.show()