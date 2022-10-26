import sys
import cv2
import torch
import numpy as np
from models.matching import Matching
from models.utils import (frame2tensor)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
random.seed(888)

torch.set_grad_enabled(False)

def get_superpoint_correspondence(frame1, frame2, matching):
    frame1_tensor = frame2tensor(frame1, device)
    frame2_tensor = frame2tensor(frame2, device)
    kp1_data = matching.superpoint({'image': frame1_tensor})
    kp2_data = matching.superpoint({'image': frame2_tensor})
    kp1 = kp1_data["keypoints"][0].cpu().numpy()
    # kp1_score = kp1_data["scores"][0].cpu().numpy()
    des1 = kp1_data["descriptors"][0].cpu().numpy()
    kp2 = kp2_data["keypoints"][0].cpu().numpy()
    # kp2_score = kp2_data["scores"][0].cpu().numpy()
    des2 = kp2_data["descriptors"][0].cpu().numpy()
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1.T, des2.T, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx] for m in good_matches])
    points2 = np.array([kp2[m.trainIdx] for m in good_matches])
    return points1, points2

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


def DLT(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, -p1[0], -p1[1], -1, p2[1] * p1[0], p2[1] * p1[1], p2[1]]
        row2 = [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, S, V = np.linalg.svd(rows)
    H = (V[-1] / V[-1][-1]).reshape(3, 3)
    return H


def get_error(points, H):
    input_p1 = np.c_[points[:, 0:2], np.ones((points.shape[0], 1))]
    target_p2 = points[:, 2:4]
    temp_pts = H @ input_p1.T
    result_pts = temp_pts / temp_pts[2]
    errors = np.linalg.norm(target_p2 - result_pts[:2].T, axis=1) ** 2
    return errors


def ransac(matches, threshold, k, iters):
    num_best_inliers = 0
    for i in range(iters):
        sampled_idx = idx = random.sample(range(len(matches)), k)
        list_point = [matches[i] for i in sampled_idx]
        points = np.array(list_point)
        H = homography(points)
        # H = DLT(points)
        if np.linalg.matrix_rank(H) < 3:
            continue
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_errors = errors
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
    plt.scatter(np.arange(len(best_errors)), best_errors)
    plt.show()
    plt.close()
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    print("{:.2f}%".format(num_best_inliers / len(matches) * 100))
    return best_inliers, best_H

def plot_combined_img(img1, img2, inliers):
    left_image = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1]))
    left_image[:img1.shape[0], :img1.shape[1]] = img1[:, :]
    right_image = np.zeros((max(img1.shape[0], img2.shape[0]), img2.shape[1]))
    right_image[:img2.shape[0], :img2.shape[1]] = img2[:, :]
    img_combine = np.concatenate((left_image, right_image), axis=1)
    for i in range(len(inliers)):
        start_point = inliers[i][:2]
        end_point = inliers[i][2:4]
        img_combine = \
            cv2.line(
                img_combine,
                (int(start_point[0]), int(start_point[1])),
                (int(end_point[0] + left_image.shape[1]), int(end_point[1])),
                (0, 0, 0), 1
            )
    plt.imshow(img_combine, cmap='gray')
    plt.show()

def Normalized_pts(pts):
    array_pts = np.array(pts)
    mean_pts = np.mean(array_pts, 0)
    std_pts = np.std(array_pts)
    restore_normalized = [
        [std_pts / np.sqrt(2), 0, mean_pts[0]],
        [0, std_pts / np.sqrt(2), mean_pts[1]],
        [0, 0, 1]
    ]
    inv_restore_normalized = np.linalg.inv(restore_normalized)
    result = inv_restore_normalized @ np.r_[array_pts.T, np.ones((1, array_pts.shape[0]))]
    return result[:2].T, inv_restore_normalized

def denormalized_mtx(trans1, trans2, normalized_H):
    return (np.linalg.pinv(trans2) @ normalized_H) @ trans1

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        }
    }
    list_k = [4, 8, 20]
    matching = Matching(config).eval().to(device)
    img1 = cv2.imread(sys.argv[1], 0)
    img2 = cv2.imread(sys.argv[2], 0)
    gt_correspondences = np.load(sys.argv[3])
    points1, points2 = get_superpoint_correspondence(img1, img2, matching)
    matches = np.c_[points1, points2]
    plot_combined_img(img1, img2, matches)
    best_inliner, H = ransac(matches, 1, 4, 100)
    for k in list_k:
        plot_combined_img(img1, img2, best_inliner[:k])
    plot_combined_img(img1, img2, best_inliner)
    input_pts = gt_correspondences[0]
    target_pts = gt_correspondences[1]
    concat_input = np.c_[input_pts, np.ones((input_pts.shape[0], 1))]

    for k in list_k:
        H = homography(best_inliner[:k])
        temp_pts = H @ concat_input.T
        result_pts = temp_pts / temp_pts[2]
        reproj_err = np.sum(np.sqrt((result_pts[:2].T - target_pts)**2)) / 100
        print("Homogrphy Estimation %s points: %s"% (k, reproj_err))

    for k in list_k:
        H = DLT(best_inliner[:k])
        temp_pts = H @ concat_input.T
        result_pts = temp_pts / temp_pts[2]
        reproj_err = np.sum(np.sqrt((result_pts[:2].T - target_pts)**2)) / 100
        print("DLT %s points: %s"% (k, reproj_err))

    norm_pts1, trans_mtx_1 = Normalized_pts(best_inliner[:, :2])
    norm_pts2, trans_mtx_2 = Normalized_pts(best_inliner[:, 2:4])
    for k in list_k:
        normalized_H = DLT(np.c_[norm_pts1, norm_pts2][:k])
        H = denormalized_mtx(trans_mtx_1, trans_mtx_2, normalized_H)
        temp_pts = H @ concat_input.T
        result_pts = temp_pts / temp_pts[2]
        reproj_err = np.sum(np.sqrt((result_pts[:2].T - target_pts)**2)) / 100
        print("Normalized DLT %s points: %s"% (k, reproj_err))
