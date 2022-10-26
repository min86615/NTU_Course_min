from distutils import extension
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R


def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
    return axes

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


def load_cam_poses(cam_rot, cam_trans):
    axes = o3d.geometry.LineSet()
    w, h, z = 0.25, 0.25, 0.25
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [w, h, z], [-w, h, z], [-w, -h, z], [w, -h, z]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]) # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])  # R, G, B
    axes.rotate(cam_rot)
    axes.translate(cam_trans)
    return axes


def calc_camera_position(rvec, tvec):
    matrix_r = R.from_rotvec(rvec.reshape(1,3)).as_matrix().reshape(3,3)
    matrix_t = tvec.reshape(3,1)
    matrix_rt = np.c_[matrix_r, matrix_t]
    extension_axis =np.array([[0, 0, 0, 1]])
    matrix_rt = np.r_[matrix_rt, extension_axis]
    matrix_rt_inv = np.linalg.inv(matrix_rt)
    cam_rot = matrix_rt_inv[:3, :3]
    cam_trans = matrix_rt_inv[:3, 3]
    return cam_rot, cam_trans


if __name__ == '__main__':
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    ## load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)

    ## load camera matrix
    list_rot = np.load('./list_rot.npy')
    list_trans = np.load('./list_trans.npy')

    ## load axes
    for idx in range(len(list_rot)):
        cam_rot, cam_trans = calc_camera_position(list_rot[idx], list_trans[idx])
        axes = load_cam_poses(cam_rot, cam_trans)
        vis.add_geometry(axes)


    ## just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)


    vis.run()
    vis.destroy_window()