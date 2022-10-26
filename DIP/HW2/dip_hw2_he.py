import cv2
import numpy as np
import matplotlib.pyplot as plt

def bgr2gray(img):
    return ( img[:, :, 0]*0.114 + img[:, :, 1]*0.587 + img[:, :, 2]*0.299).astype('uint8').reshape(img.shape[0],img.shape[1],1)
def hist_equal(img_array):
    flat = img_array.flatten()
    n = len(flat)
    img_bincount = np.bincount(flat)
    T = 190 * np.cumsum(img_bincount)/n
    T = T.astype('uint8')
    return T[img_array.astype('uint8')]
def custom_equal(img_array):
    flat = img_array.flatten()
    n = len(flat)
    img_bincount = np.bincount(flat)
    new_mapping_table = np.arange(len(img_bincount))
    T = ((new_mapping_table-73) * 0.5) + np.clip(((new_mapping_table-85) * 6.5), 0, 2222) - np.clip(((new_mapping_table-103) * 5.5), 0, 2222)
    print(T)
    T = T.astype('uint8')
    return T[img_array.astype('uint8')]
img=cv2.imread("479632_einstein-low-contrast.tif", cv2.IMREAD_GRAYSCALE)
after_hist = hist_equal(img)
# gray_img = bgr2gray(img)
array_img = np.asarray(img)
result = custom_equal(array_img)
flat = after_hist.flatten()
plt.hist(flat, bins=50)
plt.show()
plt.imshow(after_hist, cmap = 'gray')
plt.show()