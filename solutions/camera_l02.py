"""L02 参考答案（先自己敲，卡住再看）。"""

import numpy as np


def transform_points(T, pts):
    homo = np.hstack([pts, np.ones((len(pts), 1))])     # (N, 4)
    return (T @ homo.T).T[:, :3]


def project_points(K, T_cam_from_ego, pts_ego, width, height):
    pts_cam = transform_points(T_cam_from_ego, pts_ego)
    depth = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = (K @ pts_cam.T).T                          # (N, 3)：(f·x + cx·z, f·y + cy·z, z)
        uv = uvw[:, :2] / uvw[:, 2:3]
    valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, depth, valid


def pixel_to_ray(K, u, v):
    return np.linalg.inv(K) @ np.array([u, v, 1.0])
