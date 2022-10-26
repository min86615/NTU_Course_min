import os

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from P3PRANSAC import solveP3P_RANSAC

np.random.seed(1094205)

images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    retval, rvec, tvec, inliers = solveP3P_RANSAC(points3D, points2D, cameraMatrix, distCoeffs)
    # retval, rvec, tvec, inliers = cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
    return retval, rvec, tvec, inliers


def get_trans_error(gt, estimation):
    return np.linalg.norm((gt - estimation), axis=1)


def get_rot_error(gt, estimation):
    inv_rot = R.from_quat(estimation).inv().as_quat()
    difference = gt * inv_rot
    difference_rot = R.from_quat(difference).as_rotvec()
    return np.linalg.norm(difference_rot, axis=1)


# Process model descriptors
desc_df = average_desc(train_df, points3D_df)
kp_model = np.array(desc_df["XYZ"].to_list())
desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

# Load query image
idx = 200
fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

list_err_trans = []
list_err_rot = []
list_rot = []
list_trans = []

# Load query keypoints and descriptors
valid_imgid = images_df[images_df["NAME"].str.slice(0,5) == "valid"]["IMAGE_ID"]
if os.path.exists("list_rot.npy") and os.path.exists("list_trans.npy"):
    list_rot = np.load("list_rot.npy")
    list_trans = np.load("list_trans.npy")
else:
    for idx, imgid in enumerate(valid_imgid):
        print("Image processing: %s/%s" % (idx+1, len(valid_imgid)))
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==imgid]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
        list_rot.append(rvec)
        list_trans.append(tvec)
    np.save('./list_rot.npy', np.array(list_rot))
    np.save('./list_trans.npy', np.array(list_trans))


for idx, imgid in enumerate(valid_imgid):
    rotq = R.from_rotvec(list_rot[idx].reshape(1,3)).as_quat()
    tvec = list_trans[idx].reshape(1,3)

    # Get camera pose groudtruth 
    ground_truth = images_df.loc[images_df["IMAGE_ID"]==imgid]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
    tvec_gt = ground_truth[["TX","TY","TZ"]].values
    err_trans = get_trans_error(tvec_gt, tvec)
    err_rot = get_rot_error(rotq_gt, rotq)
    list_err_trans.append(err_trans)
    list_err_rot.append(err_rot)
    print("IMG: %s Error: t %s r %s " % (imgid, err_trans, err_rot))
print("Median Transition error: %s" % np.median(np.array(list_err_trans)))
print("Median Rotation error: %s" % np.median(np.array(list_err_rot)))
