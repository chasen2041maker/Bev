"""几何：相机投影、俯视图格子。

坐标系约定（整个项目统一）：
  - 车坐标系（ego）：x 向前、y 向左、z 向上，单位米。和 highway-sim-eval 一样。
  - 相机坐标系（camera）：x 向右、y 向下、z 向前（镜头指的方向），单位米。OpenCV / KITTI / nuScenes 都用这个。
  - 图像像素：u 向右、v 向下，原点在左上角。
  - 俯视图格子（BEV）：车在正中间；行号 row 向下增大（往车后），列号 col 向右增大（往车右）。
"""

from __future__ import annotations


# ---------- L01：你来敲 ----------

def project_to_image(x_c: float, y_c: float, z_c: float, f_px: float, cx: float, cy: float) -> tuple[float, float]:
    """针孔相机：相机坐标系里的一个点 (x_c, y_c, z_c) 落在照片的哪个像素 (u, v)。

    u = f · x / z + cx，v = f · y / z + cy。z ≤ 0 的点在相机后面，拍不到，抛 ValueError。
    """
    raise NotImplementedError("L01：你来敲，见 docs/L01_perspective_vs_bev.md 第 3 步")


def ego_to_bev_pixel(x_m: float, y_m: float, res_m: float = 0.512, size_px: int = 200) -> tuple[float, float]:
    """车坐标系里地面上的一点 (x 前, y 左) 落在俯视图格子的哪里 (row, col)。

    车在正中间 (size_px / 2, size_px / 2)；往前 x 米，行号减 x / res_m；往左 y 米，列号减 y / res_m。
    默认 200 × 200 格、每格 0.512 米，覆盖前后左右各 51.2 米（BEVFormer 在 nuScenes 上的设置）。
    """
    raise NotImplementedError("L01：你来敲，见 docs/L01_perspective_vs_bev.md 第 3 步")
