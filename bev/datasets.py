"""L03 数据集：读 KITTI 标注、算 3D 框角点、走 nuScenes 的“世界 → 车 → 相机”链。

KITTI 坐标（标注 location 用的）：左彩色相机校正后的相机坐标系，x 右、y 下、z 前，米。
nuScenes 坐标：global（世界）、ego（车，x 前 y 左 z 上）、sensor（相机：x 右 y 下 z 前）。
"""

from __future__ import annotations

import numpy as np

from bev.camera import transform_points
from bev.pose import pose_to_T

KITTI_FIELDS = ["type", "truncated", "occluded", "alpha",
                "left", "top", "right", "bottom",      # 2D 框（像素）
                "h", "w", "l",                         # 3D 尺寸：高、宽、长（米）——注意顺序
                "x", "y", "z",                         # 3D 位置（相机坐标系，米）
                "rotation_y"]                          # 绕相机 y 轴（向下）的朝向（弧度）


# ---------- L03：你来敲 ----------

def parse_kitti_label(line: str) -> dict:
    """KITTI 标注的一行（15 个用空格隔开的值）→ 字典。type 是字符串，occluded 是 int，其余都是 float。

    键名用 KITTI_FIELDS 里的名字。
    """
    raise NotImplementedError("L03：你来敲，见 docs/L03_datasets.md 第 4 步")


def kitti_box_corners(h: float, w: float, l: float, x: float, y: float, z: float, ry: float) -> np.ndarray:
    """KITTI 的 3D 框 → 8 个角点 (8, 3)，在相机坐标系里。

    (x, y, z) 是框的**底面中心**（不是正中心！）。框自己的坐标里：长 l 沿 x、高 h 沿 −y（往上）、宽 w 沿 z。
    先在框自己的坐标里写出 8 个角，再绕 y 轴转 ry，最后平移到 (x, y, z)。
    绕 y 轴的旋转矩阵：[[cos, 0, sin], [0, 1, 0], [−sin, 0, cos]]。
    角点顺序：前 4 个是底面（y = 0），后 4 个是顶面（y = −h），各自 x 取 [l/2, l/2, −l/2, −l/2]、z 取 [w/2, −w/2, −w/2, w/2]。
    """
    raise NotImplementedError("L03：你来敲，见 docs/L03_datasets.md 第 4 步")


def global_to_camera(pts_global: np.ndarray, ego_pose: dict, calib: dict) -> np.ndarray:
    """nuScenes：世界坐标系里的点 (N, 3) → 相机坐标系 (N, 3)。

    ego_pose 给的是 T_global_from_ego，calib（calibrated_sensor）给的是 T_ego_from_cam，都用 pose_to_T 变成矩阵。
    要的是 T_cam_from_global = inv(T_ego_from_cam) @ inv(T_global_from_ego)，再用 L02 的 transform_points。
    """
    raise NotImplementedError("L03：你来敲，见 docs/L03_datasets.md 第 4 步")
