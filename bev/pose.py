"""位姿工具：四元数 → 旋转矩阵、位姿 → 4 × 4 矩阵。助手写好的，不用手敲，但要看懂。

nuScenes 的旋转都用四元数 (w, x, y, z) 存：绕单位轴 n 转 θ 角，就是 (cos θ/2, sin θ/2 · n)。
例：绕 z 轴（向上）转 90° → (cos 45°, 0, 0, sin 45°) = (0.7071, 0, 0, 0.7071)。
"""

from __future__ import annotations

import numpy as np


def quat_to_rot(q) -> np.ndarray:
    """四元数 (w, x, y, z) → 3 × 3 旋转矩阵。"""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_T(rotation_wxyz, translation) -> np.ndarray:
    """nuScenes 风格的位姿（rotation 四元数 + translation）→ 4 × 4 矩阵。

    nuScenes 里 ego_pose 是 T_global_from_ego（车在世界里的位置朝向），
    calibrated_sensor 是 T_ego_from_sensor（传感器在车上的位置朝向）。
    """
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rotation_wxyz)
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T
