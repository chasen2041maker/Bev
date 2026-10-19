"""L03 实验：把 KITTI 和 nuScenes 的“一行行数字”画回照片上。

这个实验要回答：
  1. KITTI 标注里的 location 是框的正中心还是底面中心？投影出来和人工标的 2D 框比一比就知道。
  2. 激光雷达坐标系里，那个行人在哪？（课件 001 第 16 页的链反过来走）
  3. nuScenes 的框存在世界坐标系里，要经过哪几步才能画到前视照片上？

运行（在仓库根目录）：python -m labs.l03_read_datasets
"""

import json
import math
from pathlib import Path

import numpy as np

from bev.camera import transform_points
from bev.datasets import global_to_camera, kitti_box_corners, parse_kitti_label
from bev.pose import pose_to_T, quat_to_rot
from data.kitti_samples import IMAGE_H, IMAGE_W, LABEL_000000, parse_calib


def project_P(P, pts):
    uvw = (P @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
    return uvw[:, :2] / uvw[:, 2:3]


def box_of(uv):
    return uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()


def iou_2d(a, b):
    """两个 2D 框（left, top, right, bottom）的交并比——L04 会正式讲。"""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def main():
    calib = parse_calib()
    d = parse_kitti_label(LABEL_000000)
    label_box = (d["left"], d["top"], d["right"], d["bottom"])
    print("① KITTI 000000 的行人：", {k: d[k] for k in ("type", "h", "w", "l", "x", "y", "z", "rotation_y")})

    bottom = kitti_box_corners(d["h"], d["w"], d["l"], d["x"], d["y"], d["z"], d["rotation_y"])
    center = kitti_box_corners(d["h"], d["w"], d["l"], d["x"], d["y"] + d["h"] / 2, d["z"], d["rotation_y"])   # 假如 location 是正中心
    b_box, c_box = box_of(project_P(calib["P2"], bottom)), box_of(project_P(calib["P2"], center))
    fmt = lambda b: "(" + ", ".join(f"{v:.1f}" for v in b) + ")"
    print(f"   人工标的 2D 框        {fmt(label_box)}")
    print(f"   当成“底面中心”投影   {fmt(b_box)}  和人工框的 IoU = {iou_2d(b_box, label_box):.2f}")
    print(f"   当成“正中心”投影     {fmt(c_box)}  和人工框的 IoU = {iou_2d(c_box, label_box):.2f}")

    # ② 相机坐标 → 激光雷达坐标：课件第 16 页 y = P2 · R0_rect · Tr_velo_to_cam · x，倒着走
    T_rect_from_velo = np.eye(4)
    T_rect_from_velo[:3, :] = calib["Tr_velo_to_cam"]
    R0 = np.eye(4)
    R0[:3, :3] = calib["R0_rect"]
    T_velo_from_rect = np.linalg.inv(R0 @ T_rect_from_velo)
    p_velo = transform_points(T_velo_from_rect, np.array([[d["x"], d["y"] - d["h"] / 2, d["z"]]]))[0]
    print(f"\n② 行人中心在激光雷达坐标系（x 前 y 左 z 上）：({p_velo[0]:.2f}, {p_velo[1]:.2f}, {p_velo[2]:.2f})"
          f" → 前方约 {p_velo[0]:.1f} 米、右边约 {-p_velo[1]:.1f} 米")

    # ③ nuScenes：世界 → 车 → 相机 → 像素
    s = json.loads(Path("data/nuscenes_sample.json").read_text(encoding="utf-8"))
    ann, ego, cal = s["sample_annotation"], s["ego_pose"], s["calibrated_sensor"]
    K = np.array(cal["camera_intrinsic"])
    c_global = np.array([ann["translation"]])
    c_ego = transform_points(np.linalg.inv(pose_to_T(ego["rotation"], ego["translation"])), c_global)
    c_cam = global_to_camera(c_global, ego, cal)
    uv = (K @ c_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    print(f"\n③ nuScenes 那辆车的中心：世界 ({c_global[0, 0]:.1f}, {c_global[0, 1]:.1f}, {c_global[0, 2]:.1f}) → 车 ({c_ego[0, 0]:.2f}, {c_ego[0, 1]:.2f}, {c_ego[0, 2]:.2f})"
          f" → 相机 ({c_cam[0, 0]:.2f}, {c_cam[0, 1]:.2f}, {c_cam[0, 2]:.2f}) → 像素 ({uv[0, 0]:.1f}, {uv[0, 1]:.1f})")
    print("   前视相机的旋转四元数 (0.5, −0.5, 0.5, −0.5) 换成矩阵：")
    print("   " + str(np.round(quat_to_rot(cal["rotation"]), 3)).replace("\n", "\n   ") + "  ← 就是 L02 的 R_EGO_FROM_CAMAXES")
    plot(calib, label_box, bottom, center, s)


# ---------- 以下是画图，助手写好的，不用手敲 ----------
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def nuscenes_corners(center, size_wlh, q):
    w, l, h = size_wlh
    xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    zs = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    return (quat_to_rot(q) @ np.vstack([xs, ys, zs])).T + np.asarray(center)


def plot(calib, label_box, bottom, center, s):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("没有装 matplotlib，跳过画图。需要的话运行：python -m pip install matplotlib")
        return
    fig, (ax_k, ax_n) = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax_k.set_xlim(0, IMAGE_W); ax_k.set_ylim(IMAGE_H, 0); ax_k.set_aspect("equal")
    ax_k.add_patch(Rectangle((label_box[0], label_box[1]), label_box[2] - label_box[0], label_box[3] - label_box[1],
                             fill=False, edgecolor="#1baf7a", linewidth=2, label="2D box (human label)"))
    for corners, col, name in ((bottom, "#1e66d6", "3D box, location = bottom center"), (center, "#eb6834", "3D box, if location = center")):
        uv = project_P(calib["P2"], corners)
        for k, (i, j) in enumerate(EDGES):
            ax_k.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]], color=col, linewidth=1.5, label=name if k == 0 else None)
    ax_k.set_xlim(600, 950); ax_k.set_ylim(400, 80)
    ax_k.legend(frameon=False, fontsize=8, loc="lower right")
    ax_k.set_title("KITTI 000000: pedestrian (zoomed)")
    ann, ego, cal = s["sample_annotation"], s["ego_pose"], s["calibrated_sensor"]
    K = np.array(cal["camera_intrinsic"])
    corners = nuscenes_corners(ann["translation"], ann["size"], ann["rotation"])
    cam = global_to_camera(corners, ego, cal)
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    W, H = s["image_size"]
    ax_n.set_xlim(0, W); ax_n.set_ylim(H, 0); ax_n.set_aspect("equal")
    ax_n.axhline(K[1, 2], color="#9a9a94", linewidth=1, linestyle=":")
    for i, j in EDGES:
        ax_n.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]], color="#1e66d6", linewidth=1.5)
    ax_n.set_title("nuScenes-style sample: car 30 m ahead, CAM_FRONT")
    for ax in (ax_k, ax_n):
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = Path("outputs"); out.mkdir(exist_ok=True)
    fig.savefig(out / "l03_datasets.png", dpi=110)
    print("\n图已保存：outputs/l03_datasets.png")


if __name__ == "__main__":
    main()
