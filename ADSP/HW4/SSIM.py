import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def SSIM(img_x, img_y, c1, c2):
    mean_img_x = np.mean(img_x)
    mean_img_y = np.mean(img_y)
    Lx = np.max(img_x) - np.min(img_x)
    cov_x_sqr = np.mean((img_x - mean_img_x)**2)
    cov_y_sqr = np.mean((img_y - mean_img_y)**2)
    cov_xy = np.mean((img_x - mean_img_x)*(img_y - mean_img_y))
    return ((2*mean_img_x*mean_img_y + (c1*Lx)**2)*(2*cov_xy + (c2*Lx)**2))/ \
        ((mean_img_x**2 + mean_img_y**2 + (c1*Lx)**2)*(cov_x_sqr + cov_y_sqr + (c2*Lx)**2))
        
img_x = cv2.imread("baboon.jpg", cv2.IMREAD_GRAYSCALE)
img_y = cv2.imread("pepper.png", cv2.IMREAD_GRAYSCALE)
resize_img_x = cv2.resize(img_x, (360, 360), interpolation=cv2.INTER_CUBIC)
resize_img_y = cv2.resize(img_y, (360, 360), interpolation=cv2.INTER_CUBIC)
resize_img_x_lighten = np.clip(resize_img_x.astype(float)+100, 0, 255).astype('uint8')
c1 = c2 = (1/255)**0.5
SSIM_different = SSIM(resize_img_x, resize_img_y, c1, c2)
SSIM_lighten = SSIM(resize_img_x, resize_img_x_lighten, c1, c2)
f, axs = plt.subplots(3,1,figsize=(10,10))
axs[0].axis("off")
axs[0].imshow(resize_img_x, cmap="gray")
axs[1].set_title("SSIM = %s" % SSIM_different)
axs[1].imshow(resize_img_y, cmap="gray")
axs[2].set_title("SSIM = %s" % SSIM_lighten)
axs[2].imshow(resize_img_x_lighten, cmap="gray")
plt.savefig("Comparison.png")
# print("SSIM\ndiff img: %s\nlighten img: %s"%(SSIM_different, SSIM_lighten))