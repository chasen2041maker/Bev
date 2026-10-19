"""L02 相机模型：齐次坐标、外参 T、内参 K、投影和反投影（numpy 版，一次处理 N 个点）。

坐标系约定见 bev/geometry.py 开头：车坐标 x 前 y 左 z 上；相机坐标 x 右 y 下 z 前；像素 u 右 v 下。
矩阵命名：T_a_from_b 把 b 坐标系里的点变到 a 坐标系里（4 × 4）。读法：从右往左，“从 b 到 a”。
"""

from __future__ import annotations

import numpy as np


# ---------- L02：你来敲 ----------

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """用 4 × 4 齐次矩阵 T 变换 N 个点。pts 形状 (N, 3)，返回 (N, 3)。

    做法：每个点后面补一个 1 变成 (N, 4)，乘上 T（注意 T 作用在列向量上），再去掉最后一列。
    """
    raise NotImplementedError("L02：你来敲，见 docs/L02_camera_model.md 第 4 步")


def project_points(K: np.ndarray, T_cam_from_ego: np.ndarray, pts_ego: np.ndarray,
                   width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """车坐标系里的 N 个点 → 某个相机的像素。

    返回 (uv, depth, valid)：
      uv     (N, 2)：像素坐标 u, v
      depth  (N,)  ：点在相机坐标系里的 z（离镜头多远，沿镜头方向）
      valid  (N,)  ：True 表示这个相机真的拍得到——在镜头前面（depth > 0）而且落在照片里（0 ≤ u < width, 0 ≤ v < height）
    depth ≤ 0 的点 uv 没有意义，但也照样算出来（不报错），靠 valid 区分。
    """
    raise NotImplementedError("L02：你来敲，见 docs/L02_camera_model.md 第 4 步")


def pixel_to_ray(K: np.ndarray, u: float, v: float) -> np.ndarray:
    """反过来：一个像素对应相机坐标系里的哪条射线？返回方向 (x, y, 1)，即 z = 1 处的那个点。

    这条射线上每个点都投到同一个像素——所以只看照片，深度是丢掉的。乘上深度 d 就得到那个 3D 点。
    """
    raise NotImplementedError("L02：你来敲，见 docs/L02_camera_model.md 第 4 步")
