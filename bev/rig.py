"""相机支架（6 个相机，接近 nuScenes 的布置）。助手写好的，不用手敲，但要看懂：后面 DETR3D、BEVFormer 都用它。"""

from __future__ import annotations

import math

import numpy as np

# 相机坐标轴在车坐标系里的方向（列向量）：相机 x（右）= 车 −y，相机 y（下）= 车 −z，相机 z（前）= 车 +x
R_EGO_FROM_CAMAXES = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])


def rot_z(yaw_rad: float) -> np.ndarray:
    """绕 z 轴（向上）转 yaw，逆时针（往左）为正。"""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_T(R: np.ndarray, t) -> np.ndarray:
    """旋转 R (3 × 3) + 平移 t (3,) 拼成 4 × 4 齐次矩阵。"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def camera_extrinsics(x_m: float, y_m: float, z_m: float, yaw_deg: float) -> np.ndarray:
    """装在车上 (x, y, z)、镜头朝 yaw_deg 方向（0 = 正前，90 = 正左）的相机：返回 T_cam_from_ego。

    先写出 T_ego_from_cam（相机在车里的位置和朝向，好写），再求逆。
    """
    R_ego_from_cam = rot_z(math.radians(yaw_deg)) @ R_EGO_FROM_CAMAXES
    T_ego_from_cam = make_T(R_ego_from_cam, [x_m, y_m, z_m])
    return np.linalg.inv(T_ego_from_cam)


def make_K(f_px: float, cx: float, cy: float) -> np.ndarray:
    """内参矩阵（这里 fx = fy = f，没有倾斜）。"""
    return np.array([[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]])


WIDTH, HEIGHT = 1600, 900
K_DEFAULT = make_K(1000.0, 800.0, 450.0)   # 视场角 = 2·atan(800 / 1000) ≈ 77°；nuScenes 前视约 70°（f ≈ 1266）

# (名字, x, y, z, yaw°)：位置和朝向接近 nuScenes 的 6 个相机（数值取整，方便手算）
RIG = [
    ("CAM_FRONT", 1.7, 0.0, 1.5, 0.0),
    ("CAM_FRONT_LEFT", 1.5, 0.5, 1.5, 55.0),
    ("CAM_FRONT_RIGHT", 1.5, -0.5, 1.5, -55.0),
    ("CAM_BACK_LEFT", 1.0, 0.5, 1.5, 110.0),
    ("CAM_BACK_RIGHT", 1.0, -0.5, 1.5, -110.0),
    ("CAM_BACK", 0.0, 0.0, 1.5, 180.0),
]


def rig_extrinsics():
    """返回 {相机名: T_cam_from_ego}。"""
    return {name: camera_extrinsics(x, y, z, yaw) for name, x, y, z, yaw in RIG}
