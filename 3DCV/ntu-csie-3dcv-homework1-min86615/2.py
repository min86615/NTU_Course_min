import cv2
import numpy as np
import math

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1], -p2[1]]
        row2 = [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, S, V = np.linalg.svd(rows)
    H = (V[-1] / V[-1][-1]).reshape(3, 3)
    return H


def bilinear_interp(pts, img):
    y_floor = math.floor(pts[1])
    y_ceil = math.ceil(pts[1])
    x_floor = math.floor(pts[0])
    x_ceil = math.ceil(pts[0])
    a = pts[1] - y_floor
    b = pts[0] - x_floor
    rgb_val = np.zeros(3)
    for i in range(len(rgb_val)):
        channel_val = \
            a * b * (img[y_ceil, x_ceil, i]) + \
            a * (1 - b) * (img[y_ceil, x_floor, i]) + \
            (1 - a) * b * (img[y_floor, x_ceil, i]) + \
            (1 - a) * (1 - b) * (img[y_floor, x_floor, i])
        rgb_val[i] = channel_val
    return rgb_val
raw_image = cv2.imread("images/IMG_6385_resized.jpg")
selected_pts = np.array([[135, 261], [82, 562], [651, 569], [613, 269]]).astype("float")
shape_size = (420, 410, 3)
warp_img = np.zeros(shape_size)
tran_coor = \
    np.array(
        [
            [0, 0],
            [shape_size[0] - 1, 0],
            [shape_size[0] - 1, shape_size[1] - 1],
            [0, shape_size[1] - 1]
        ]
    ).astype("float")

H = homography(np.c_[selected_pts, tran_coor])
H_inv = np.linalg.inv(H)
for i in range(shape_size[0]):
    for j in range(shape_size[1]):
        temp_pts = H_inv @ np.array([[i, j, 1]]).T
        result_pts = (temp_pts / temp_pts[2])[:2]
        rgb_val = bilinear_interp(result_pts, raw_image)
        warp_img[i, j, :] = rgb_val


cv2.imshow("main_wd", warp_img.astype("uint8"))
cv2.waitKey(0)
cv2.destroyAllWindows()
