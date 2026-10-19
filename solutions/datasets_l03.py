"""L03 参考答案（先自己敲，卡住再看）。"""

import numpy as np

from bev.camera import transform_points
from bev.datasets import KITTI_FIELDS
from bev.pose import pose_to_T


def parse_kitti_label(line):
    vals = line.split()
    out = {}
    for key, v in zip(KITTI_FIELDS, vals):
        if key == "type":
            out[key] = v
        elif key == "occluded":
            out[key] = int(v)
        else:
            out[key] = float(v)
    return out


def kitti_box_corners(h, w, l, x, y, z, ry):
    xs = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    ys = [0, 0, 0, 0, -h, -h, -h, -h]
    zs = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (R @ np.array([xs, ys, zs])).T + np.array([x, y, z])


def global_to_camera(pts_global, ego_pose, calib):
    T_global_from_ego = pose_to_T(ego_pose["rotation"], ego_pose["translation"])
    T_ego_from_cam = pose_to_T(calib["rotation"], calib["translation"])
    T_cam_from_global = np.linalg.inv(T_ego_from_cam) @ np.linalg.inv(T_global_from_ego)
    return transform_points(T_cam_from_global, pts_global)
