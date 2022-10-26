import argparse
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from SIFT_Method import SIFT
from util import BFmatcher, cylinder_proj, get_image_shift, stitching_image


def seq_read_image_SIFT(
        list_image_path, sift_num_intervals=2, sift_contrast_threshold=0.15,
        focal_length=1600, resize_ratio=1):
    list_cylinder_images = []
    # list_gray_images = []
    list_kp = []
    list_des = []
    for idx, image_path in enumerate(list_image_path):
        print("Process: %s/%s images" % (idx + 1, len(list_image_path)))
        raw_image = cv2.imread(image_path)
        raw_image = cv2.resize(
            raw_image,
            (raw_image.shape[1] // resize_ratio, raw_image.shape[0] // resize_ratio),
            interpolation=cv2.INTER_AREA)
        # print(raw_image.shape)
        cylinder_image = cylinder_proj(raw_image, focal_length)
        gray_cylinder_image = cv2.cvtColor(cylinder_image, cv2.COLOR_BGR2GRAY)
        img_kp, img_des = SIFT(
            gray_cylinder_image,
            num_intervals=sift_num_intervals,
            contrast_threshold=sift_contrast_threshold).SIFTDetectCompute()
        list_cylinder_images.append(cylinder_image)
        # list_gray_images.append(gray_cylinder_image)
        list_kp.append(img_kp)
        list_des.append(img_des)
    return list_cylinder_images, list_kp, list_des

def seq_get_image_shift(list_cylinder_images, list_kp, list_des):
    list_image_shift = []
    for i in range(len(list_cylinder_images) - 1):
        result_matches, result_kp1_loc, result_kp2_loc = \
            BFmatcher(list_des[i], list_kp[i], list_des[i + 1], list_kp[i + 1], threshold=0.6)
        image_shift = get_image_shift(result_kp1_loc, result_kp2_loc)
        list_image_shift.append(image_shift)
    image_shift_modified = np.array(list_image_shift)
    image_shift_modified[:, 0] = list_cylinder_images[0].shape[1] - image_shift_modified[:, 0]
    image_shift_modified[:, 1] = image_shift_modified[:, 1]
    # find upper and lower bond
    list_img_y_loc = np.zeros((image_shift_modified.shape[0] + 1, image_shift_modified.shape[1]))
    for i in range(len(list_img_y_loc)):
        if i == 0:
            list_img_y_loc[i, 0] = 0
            list_img_y_loc[i, 1] = list_cylinder_images[0].shape[0]
        else:
            list_img_y_loc[i, :] = list_img_y_loc[i - 1, :] + image_shift_modified[i - 1, 1]
    modified_list_img_y_loc = (list_img_y_loc - (np.min(list_img_y_loc[:, 0]))).astype('int')
    return image_shift_modified, modified_list_img_y_loc

parser = argparse.ArgumentParser()
parser.add_argument(
    "--img_folder", type=str, default="../data/",
    help="Please input your image folder location")
parser.add_argument(
    "--num_intervals", type=int, default=2,
    help="Please input number of interval in SIFT implement")
parser.add_argument(
    "--contrast_threshold", type=float, default=0.08,
    help="Please input SIFT key point detection contrast threshold")
parser.add_argument(
    "--focal_length", type=int, default=1000,
    help="Please input cylinder projection focal length")
parser.add_argument(
    "--resize_ratio", type=int, default=1,
    help="If your images are too big you can set resize ration to resize your pictures")
parser.add_argument(
    "--crop", type=bool, default=True,
    help="Whether crop image after image stitching")
args = parser.parse_args()
# cylinder projection -> SIFT -> BFMatch -> get image shift -> stitch image
list_image_path = glob.glob("%s*.jpg" % (args.img_folder))
list_cylinder_images, list_kp, list_des = seq_read_image_SIFT(
    list_image_path,
    sift_num_intervals=args.num_intervals,
    sift_contrast_threshold=args.contrast_threshold,
    focal_length=args.focal_length,
    resize_ratio=args.resize_ratio)
image_shift_modified, modified_list_img_y_loc = seq_get_image_shift(list_cylinder_images, list_kp, list_des)
result = stitching_image(list_cylinder_images, image_shift_modified, modified_list_img_y_loc, crop=args.crop)
cv2.imwrite("../image_stitching.png", result)
