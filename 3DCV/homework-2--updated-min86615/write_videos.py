import os

import cv2
import glob
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def transform_vertices(id_surface, vertice_0, vertice_1, vertice_2, trans_rt):
    num_depth = 8
    list_pts = []
    vec_0 = vertice_2 - vertice_0
    vec_1 = vertice_1 - vertice_0
    
    # plot cube
    for idx in range(num_depth):
        init_point = vertice_0 + vec_0 * idx / (num_depth - 1)
        for idx2 in range(num_depth):
            point = init_point + vec_1 * idx2 / (num_depth - 1)
            trans_point = trans_rt @ np.append(point, 1.)
            trans_point /= trans_point[-1]
            list_pts.append(([int(trans_point[0]), int(trans_point[1])], point[-1], id_surface))
    return list_pts


#load features info
images_df = pd.read_pickle("data/images.pkl")

#load cube info
cube_vertices = np.load("cube_vertices.npy")
cube_transform = np.load("cube_transform_mat.npy")

idx_surface = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5],
    [2, 3, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7]
]

cameraMatrix = np.array([[1868.27, 0, 540],[0, 1869.18, 960],[0, 0, 1]])
list_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

list_frame = []
list_frame_path = glob.glob("data/frames/valid_*.jpg")
list_frame_path.sort(key=lambda x: int(x.split("_")[-1][3:-4]))
for frame_path in list_frame_path:
    img = cv2.imread(frame_path)
    h, w, channel = img.shape
    ground_truth = images_df.loc[images_df["NAME"] == frame_path.split("/")[-1]]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
    tvec_gt = ground_truth[["TX","TY","TZ"]].values
    rot_matrix = R.from_quat(rotq_gt).as_matrix().squeeze()
    trans_rt = cameraMatrix @ np.c_[rot_matrix, tvec_gt.reshape(3, 1)]
    list_plot_pts = []
    for idx, surfaceidx in enumerate(idx_surface):
        list_plot_pts += transform_vertices(idx, cube_vertices[surfaceidx[0]], cube_vertices[surfaceidx[1]], cube_vertices[surfaceidx[2]], trans_rt)
    list_plot_pts.sort(key=lambda x: -x[1])
    for (loc_point, depth, surface_idx) in list_plot_pts:
        cv2.circle(img, tuple(loc_point), 2, list_color[surface_idx], 6)
    list_frame.append(img)
video_out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 10, (w, h))
for frame in list_frame:
    # writing to a image array
    video_out.write(frame)
video_out.release()
print("BP")
# for idx, imgid in enumerate(valid_imgid):