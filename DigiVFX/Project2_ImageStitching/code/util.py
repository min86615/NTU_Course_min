import numpy as np
from scipy.spatial.distance import cdist

def cylinder_proj(img, focal_val=1000):
    h, w = img.shape[:2]
    project_img = np.zeros(img.shape)
    center_x, center_y = w // 2, h // 2
    max_pt = 0
    min_pt = img.shape[1]
    for y in range(h):
        for x in range(w):
            theta_deg = np.arctan((x - center_x) / focal_val)
            pt_x = int(focal_val * np.tan((x - center_x) / focal_val) + center_x)
            pt_y = int((y - center_y) / np.cos(theta_deg) + center_y)
            if (w > pt_x) & (pt_x >= 0) & (h > pt_y) & (pt_y >= 0):
                max_pt = x if x > max_pt else max_pt
                min_pt = x if min_pt > x else min_pt
                project_img[y, x, :] = img[pt_y, pt_x, :]
    project_img = project_img[:, min_pt:max_pt + 1, :]
    return project_img.astype("uint8")

def BFmatcher(des1, kp1, des2, kp2, threshold=0.8):
    distances = cdist(des1, des2, metric='euclidean')
    sorted_idx = np.argsort(distances, axis=1)
    # if threshold * least value > second small value accept it
    good_matches = []
    good_matches_kp1_loc = []
    good_matches_kp2_loc = []
    for i in range(len(distances)):
        if (distances[i, sorted_idx[i][1]] * threshold) > (distances[i, sorted_idx[i][0]]):
            good_matches.append([i, sorted_idx[i][0]])
            good_matches_kp1_loc.append(kp1[i].pt)
            good_matches_kp2_loc.append(kp2[sorted_idx[i][0]].pt)
    return good_matches, np.array(good_matches_kp1_loc), np.array(good_matches_kp2_loc)

def get_image_shift(list_kp1_loc, list_kp2_loc, sample_num=12, iter=100):
    list_err, list_dxy = [], []
    for idx_time in range(iter):
        samples = np.random.randint(0, len(list_kp1_loc), sample_num)
        dxy = np.mean(list_kp1_loc[samples] - list_kp2_loc[samples], axis=0).astype(np.int)
        diff_xy = np.abs(list_kp1_loc - (list_kp2_loc + dxy))
        err = np.mean(np.sum(diff_xy, axis=1))
        list_err.append(err)
        list_dxy.append(dxy)
    min_error_idx = np.argmin(list_err)
    return list_dxy[min_error_idx]

def stitching_image(list_cylinder_images, image_shift_modified, modified_list_img_y_loc, crop=True):
    base_img_height = int(np.max(modified_list_img_y_loc[:, 1]))
    for i in range(len(modified_list_img_y_loc)):
        y_low = modified_list_img_y_loc[i, 0]
        y_up = modified_list_img_y_loc[i, 1]
        if i == 0:
            concat_image = \
                np.zeros((base_img_height, list_cylinder_images[i].shape[1], list_cylinder_images[i].shape[2]))
            concat_image[y_low:y_up, :, :] = list_cylinder_images[i]
            combined_image = concat_image
        else:
            blend_ratio_2 = (np.arange(image_shift_modified[i - 1, 0]) + 1) / (image_shift_modified[i - 1, 0] + 1)
            blend_ratio_1 = blend_ratio_2[::-1]
            concat_image = \
                np.zeros((base_img_height, list_cylinder_images[i].shape[1], list_cylinder_images[i].shape[2]))
            concat_image[y_low:y_up, :, :] = list_cylinder_images[i]
            concat_image[:, :image_shift_modified[i - 1, 0], 0] *= blend_ratio_2
            concat_image[:, :image_shift_modified[i - 1, 0], 1] *= blend_ratio_2
            concat_image[:, :image_shift_modified[i - 1, 0], 2] *= blend_ratio_2
            combined_image[:, -image_shift_modified[i - 1, 0]:, 0] *= blend_ratio_1
            combined_image[:, -image_shift_modified[i - 1, 0]:, 1] *= blend_ratio_1
            combined_image[:, -image_shift_modified[i - 1, 0]:, 2] *= blend_ratio_1
            combined_image[:, -image_shift_modified[i - 1, 0]:, :] += \
                concat_image[:, :image_shift_modified[i - 1, 0], :]
            combined_image = \
                np.concatenate((combined_image, concat_image[:, image_shift_modified[i - 1, 0]:, :]), axis=1)
    if crop:
        crop_lower = int(np.max(modified_list_img_y_loc[:, 0]))
        crop_upper = int(np.min(modified_list_img_y_loc[:, 1]))
        cropped_combined_image = combined_image[crop_lower:crop_upper, :, :]
        return cropped_combined_image
    else:
        return combined_image
