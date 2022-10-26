import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def cosine(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# implement cv2.solvePnPRansac

# reference https://github.com/akshayb6/trilateration-in-3d/blob/master/trilateration.py
def solve_trilaterate(p1, p2, p3, r1, r2, r3):
    unit_v_x = (p2 - p1) / np.linalg.norm(p2 - p1)
    project_len_x = np.dot(unit_v_x, (p3 - p1))
    # project_len_x * unit_v_x -> new coordinate x direction in cartesian coordinate
    unit_v_y = (p3 - p1 - (project_len_x * unit_v_x)) / np.linalg.norm(p3 - p1 - (project_len_x * unit_v_x))
    unit_v_z = np.cross(unit_v_x, unit_v_y)
    dist_p2p1 = np.linalg.norm(p2 - p1)
    project_len_y = np.dot(unit_v_y, (p3 - p1))
    x = ((r1**2) - (r2**2) + (dist_p2p1**2)) / (2 * dist_p2p1)
    y = \
        (((r1**2) - (r3**2) + (project_len_x**2) + (project_len_y**2)) / (2 * project_len_y)) - \
        ((project_len_x / project_len_y) * x)
    z = np.sqrt(r1**2 - x**2 - y**2)
    ans_1 = p1 + (x * unit_v_x) + (y * unit_v_y) + (z * unit_v_z)
    ans_2 = p1 + (x * unit_v_x) + (y * unit_v_y) - (z * unit_v_z)
    return ans_1, ans_2
    

def solve_length(points3D, recovered_points2D):
    
    R_ab = np.linalg.norm(points3D[0] - points3D[1])
    R_bc = np.linalg.norm(points3D[1] - points3D[2])
    R_ac = np.linalg.norm(points3D[0] - points3D[2])
    
    # x1-x2 ab x1-x3 ac x2-x3 bc
    K_1 = (R_bc / R_ac) ** 2
    K_2 = (R_bc / R_ab) ** 2
    C_ab = cosine(recovered_points2D[0], recovered_points2D[1])
    C_ac = cosine(recovered_points2D[0], recovered_points2D[2])
    C_bc = cosine(recovered_points2D[1], recovered_points2D[2])

    G_4 = \
        (K_1 * K_2 - K_1 - K_2) ** 2 - \
        4 * K_1 * K_2 * (C_bc ** 2)
    G_3 = \
        4 * (K_1 * K_2 - K_1 - K_2) * K_2 * (1 - K_1) * C_ab + \
        4 * K_1 * C_bc * ((K_1 * K_2 - K_1 + K_2) * C_ac + 2 * K_2 * C_ab * C_bc)
    G_2 = \
        (2 * K_2 * (1 - K_1) * C_ab) ** 2 + \
        2 * (K_1 * K_2 - K_1 - K_2) * (K_1 * K_2 + K_1 - K_2) + \
        4 * K_1 * ((K_1 - K_2) * (C_bc ** 2) + K_1 * (1 - K_2) * (C_ac ** 2) - 2 * (1 + K_1) * K_2 * C_ab * C_ac * C_bc) 
    G_1 = \
        4 * (K_1 * K_2 + K_1 - K_2) * K_2 * (1- K_1) * C_ab + \
        4 * K_1 * ((K_1 * K_2 - K_1 + K_2) * C_ac * C_bc + 2 * K_1 * K_2 * C_ab * (C_ac ** 2))
    G_0 = \
        (K_1 * K_2 + K_1 - K_2) ** 2 - \
        4 * (K_1 ** 2) * K_2 * (C_ac ** 2)
    # in companion matrix firs row component is zero will cause rank reduce 1
    if G_4 == 0:
        return
    else:
        solve_roots = np.roots(np.array([G_4, G_3, G_2, G_1, G_0]))
    roots_real = np.real(solve_roots[np.isreal(solve_roots)])

    a = np.sqrt((R_ab**2) / (1 + roots_real**2 - 2 * roots_real * C_ab))
    b = roots_real * a
    m = 1 - K_1
    p = 2 * (K_1 * C_ac - roots_real * C_bc)
    q = roots_real ** 2 - K_1
    m_prime = 1
    p_prime = 2 * (-roots_real * C_bc)
    q_prime = (roots_real ** 2) * (1 - K_2) + 2 * roots_real * K_2 * C_ab - K_2
    y = -(m_prime * q - m * q_prime) / (p * m_prime - p_prime * m)
    c = y * a
    return a, b, c

def solve_R_T(selected_points3D, selected_points2D, val_points3D, val_points2D, a, b, c):
    list_centers = []
    for idx in range(len(a)):
        ans1, ans2 = \
            solve_trilaterate(
                selected_points3D[0],
                selected_points3D[1],
                selected_points3D[2],
                a[idx], b[idx], c[idx])
        list_centers.append(ans1)
        list_centers.append(ans2)

    list_R = []
    list_T = []
    best_R = []
    best_T = []
    tmp_min_loss = np.inf
    for center in list_centers:
        for direct in [1, -1]:
            val_lambda = direct * np.linalg.norm(selected_points3D - center, axis=1) / np.linalg.norm(selected_points2D, axis=1)
            R_val = (selected_points2D.T * val_lambda) @ np.linalg.pinv(selected_points3D.T -center.reshape(3, 1))
            # list_R.append(R_val)
            # list_T.append(center)
            reproject = (R_val @ (val_points3D.T- center.reshape(3,1)))
            reproject = (reproject / reproject[2]).T
            loss = np.linalg.norm(reproject[:, :2] - val_points2D[:, :2], axis=1).sum()
            if tmp_min_loss > loss:
                tmp_min_loss = loss
                best_R = R_val
                best_T = center
            # norm(project - undistort_points2D[:, :2][np.newaxis], axis=-1)
    # print(tmp_min_loss)
    return best_R, best_T

def solveP3P(points3D, recovered_points2D):
    # for i in range(500):
    #     recovered_points2D = np.c_[cv2.undistortPoints(points2D, cameraMatrix, distCoeffs).squeeze(), np.ones((len(points2D), 1))]
    #     selected_points_idx = np.random.randint(len(points3D), size=6)
    selected_points3D = points3D[:3]
    selected_points2D = recovered_points2D[:3]
    val_points3D = points3D[3:]
    val_points2D = recovered_points2D[3:]
    try:
        a, b, c = solve_length(selected_points3D, selected_points2D)
        best_R, best_T = solve_R_T(selected_points3D, selected_points2D, val_points3D, val_points2D, a, b, c)
        #     point = best_R@(points3D.T-best_T.reshape(3,1))
        #     point = (point / point[2])
        #     point = point[:2].T
        #     sum_error = np.linalg.norm((recovered_points2D[:, :2] - point), axis = 1).sum()
        #     print("Total: %s" % sum_error)
        # print(best_R, best_T)
        return best_R, best_T
    except:
        return [], []


def check_inliers(points3D, recovered_points2D, tmp_R, tmp_T, threshold_err):
    reproject_points = tmp_R @ (points3D.T - tmp_T.reshape(3,1))
    reproject_points = (reproject_points / reproject_points[2]).T
    list_errors = np.linalg.norm((recovered_points2D[:, :2] - reproject_points[:, :2]), axis = 1)
    inliers = np.where(threshold_err > list_errors)[0]
    return len(inliers), inliers
    


def solveP3P_RANSAC(points3D, points2D, cameraMatrix, distCoeffs):
    prob = 0.99
    err_ratio = 0.5
    take_pts = 6
    max_iter_times = int(np.ceil(np.log(1 - prob)) / (np.log(1 - (1 - err_ratio) ** (take_pts))))
    iter_times = 0
    recovered_points2D = np.c_[cv2.undistortPoints(points2D, cameraMatrix, distCoeffs).squeeze(), np.ones((len(points2D), 1))]
    status_flag = True
    threshold_d = 1e-3
    best_R, best_T = [], []
    max_inliers = 0
    max_inliers_idx = []
    # while status_flag:
    while status_flag:
        selected_points_idx = np.random.randint(len(points3D), size=take_pts)
        tmp_R, tmp_T = solveP3P(points3D[selected_points_idx], recovered_points2D[selected_points_idx])
        if len(tmp_R) > 0:
            len_inliers, list_inliers = check_inliers(points3D, recovered_points2D, tmp_R, tmp_T, threshold_d)
            if len_inliers > max_inliers:
                max_inliers = len_inliers
                max_inliers_idx = list_inliers
                selected_inliers = points3D[max_inliers_idx]
                best_R, best_T = tmp_R, tmp_T
        iter_times += 1
        if iter_times == max_iter_times:
            status_flag = False
    
    return True, R.from_matrix(best_R).as_rotvec(), -best_R @ best_T, selected_inliers
