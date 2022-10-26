import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from numpy.lib.function_base import sinc

def imgfilter2D(img, filter, ratio):
    img_x, img_y = img.shape
    filter_x, filter_y = filter.shape
    result_mtx = np.zeros(((img_x - filter_x + 1), (img_y - filter_y + 1)))
    for i in range(result_mtx.shape[0]):
        for j in range(result_mtx.shape[1]):
            result_mtx[i][j] = np.abs(np.sum(img[i:i + filter_x, j:j + filter_y] * filter) / ratio)
    return result_mtx.astype('uint8').reshape(result_mtx.shape[0], result_mtx.shape[1])

def low_pass_filter(dia, img_shape):
    filter = np.zeros(img_shape[:2])
    img_x, img_y = filter.shape
    center = (img_x//2, img_y//2)
    for x_loc in range(img_x):
        for y_loc in range(img_y):
            if dia > ((x_loc-center[0])**2 + (y_loc-center[1])**2)**0.5:
                filter[x_loc][y_loc] = 1
    return filter

def low_pass_gaussian_filter(filter_size, sigma, img_shape):
    filter = np.zeros(img_shape[:2])
    img_x, img_y = filter.shape
    center = (img_x//2, img_y//2)
    for x_loc in range(img_x):
        for y_loc in range(img_y):
            dist_x_center = x_loc-center[0]
            dist_y_center = y_loc-center[1]
            if (filter_size > abs(dist_x_center)) & (filter_size > abs(dist_y_center)):
                r_sqr = ((dist_x_center)**2 + (dist_y_center)**2)
                filter[x_loc][y_loc] = np.exp(-r_sqr/(2*(sigma**2)))
    return filter
def gaussian_filter(filter_size, sigma):
    filter = np.zeros((filter_size, filter_size))
    img_x, img_y = filter.shape
    center = (img_x//2, img_y//2)
    for x_loc in range(img_x):
        for y_loc in range(img_y):
            dist_x_center = x_loc-center[0]
            dist_y_center = y_loc-center[1]
            r_sqr = ((dist_x_center)**2 + (dist_y_center)**2)
            filter[x_loc][y_loc] = np.exp(-r_sqr/(2*(sigma**2)))
    return filter

def rever_fft_with_filter(fftshift_img, filter_size, sigma=None):
    if sigma == None:
        passfilter_fftshift_img = fftshift_img * low_pass_filter(filter_size, (fftshift_img.shape)) 
    else:
        passfilter_fftshift_img = fftshift_img * low_pass_gaussian_filter(filter_size, sigma, (fftshift_img.shape)) 
    passfilter_ifftshift_img = np.fft.ifftshift(passfilter_fftshift_img)
    passfilter_img = np.fft.ifft2(passfilter_ifftshift_img)
    return passfilter_img

img=cv2.imread("479632_checkerboard1024-shaded.tif", cv2.IMREAD_GRAYSCALE)
# padding_img = np.pad(img, (128, 128), 'reflect')
# g_filter = gaussian_filter(257, 64)
# print(np.sum(g_filter))
# gaussian_blur_img=cv2.imread("HW2_mask_gaussian.png", cv2.IMREAD_GRAYSCALE)
# gaussian_blur_img = imgfilter2D(padding_img, g_filter, np.sum(g_filter))
# cv2.imwrite('Converted.png', gaussian_blur_img)

fft_img = np.fft.fft2(img)
fftshift_img = np.fft.fftshift(fft_img)
low_pass_8 = np.abs(rever_fft_with_filter(fftshift_img, 8))
low_pass_img = np.clip(np.abs(rever_fft_with_filter(fftshift_img, 8, 4)), 0, 255)
process_img = (img/low_pass_img).astype("uint8")
plt.imshow(low_pass_img, "gray")
plt.show()
plt.imshow(process_img, "gray")
plt.show()


# img = cv2.imread('479632_checkerboard1024-shaded.tif')
# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Image', grayImg)
# cv2.waitKey(0)
# filtersize = 257
# gaussianImg = cv2.GaussianBlur(grayImg, (filtersize, filtersize), 64)
# cv2.imshow('Converted Image', gaussianImg)
# cv2.waitKey(0)
# for i in range(gaussianImg.shape[0]):
#     for j in range(gaussianImg.shape[1]):
#         gaussianImg[i][j] *= ((np.clip((1-((i**2+j**2)**(0.5))/1024), 0, 1)))
# newImg = (grayImg-gaussianImg)
# cv2.imshow('New Image', newImg)
# cv2.imwrite('Converted.png', newImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()