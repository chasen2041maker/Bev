"""L03 测试：KITTI 训练集 000000 / 000001 的真实标注 + 按 nuScenes 字段手造的样例（数字能手算）。"""

import json
import math
from pathlib import Path

import numpy as np

from bev.datasets import global_to_camera, kitti_box_corners, parse_kitti_label
from data.kitti_samples import LABEL_000000, LABEL_000001, parse_calib


def test_parse_pedestrian_line():
    """000000 那一行：行人，2D 框 (712.40, 143.00, 810.73, 307.92)，高 1.89 宽 0.48 长 1.20，位置 (1.84, 1.47, 8.41)。"""
    d = parse_kitti_label(LABEL_000000)
    assert d["type"] == "Pedestrian" and d["occluded"] == 0
    assert (d["left"], d["top"], d["right"], d["bottom"]) == (712.40, 143.00, 810.73, 307.92)
    assert (d["h"], d["w"], d["l"]) == (1.89, 0.48, 1.20)
    assert (d["x"], d["y"], d["z"], d["rotation_y"]) == (1.84, 1.47, 8.41, 0.01)


def test_parse_dontcare_and_cyclist():
    """DontCare（不评测的区域）的 3D 值都是 −1 / −1000 / −10；骑车人 occluded = 3（遮挡情况未知）。"""
    lines = [parse_kitti_label(s) for s in LABEL_000001.splitlines()]
    assert [d["type"] for d in lines] == ["Truck", "Car", "Cyclist", "DontCare"]
    assert lines[2]["occluded"] == 3
    assert lines[3]["x"] == -1000.0 and lines[3]["rotation_y"] == -10.0


def test_corners_no_rotation():
    """高 2、宽 1、长 4，底面中心 (0, 0, 10)，ry = 0：x 在 ±2，y 在 0（底）和 −2（顶），z 在 9.5 和 10.5。"""
    c = kitti_box_corners(2.0, 1.0, 4.0, 0.0, 0.0, 10.0, 0.0)
    assert c.shape == (8, 3)
    assert np.allclose(sorted(set(np.round(c[:, 0], 6))), [-2.0, 2.0])
    assert np.allclose(c[:4, 1], 0.0) and np.allclose(c[4:, 1], -2.0)
    assert np.allclose(sorted(set(np.round(c[:, 2], 6))), [9.5, 10.5])


def test_corners_rotated_90():
    """ry = π/2：车身长轴从“朝右”转成“朝前后”，所以 x 只剩 ±0.5（宽的一半），z 在 8 到 12（长 4）。"""
    c = kitti_box_corners(2.0, 1.0, 4.0, 0.0, 0.0, 10.0, math.pi / 2)
    assert np.allclose(sorted(set(np.round(c[:, 0], 6))), [-0.5, 0.5])
    assert np.allclose(sorted(set(np.round(c[:, 2], 6))), [8.0, 12.0])


def test_pedestrian_projects_near_its_2d_box():
    """用 P2 把行人的 8 个角点投到照片上，包住它们的框是 (718.1, 135.7, 830.2, 302.7)；
    人工标的 2D 框是 (712.4, 143.0, 810.7, 307.9)——差 6～20 像素，因为 2D 框是在照片上单独标的。"""
    d = parse_kitti_label(LABEL_000000)
    P2 = parse_calib()["P2"]
    c = kitti_box_corners(d["h"], d["w"], d["l"], d["x"], d["y"], d["z"], d["rotation_y"])
    uvw = (P2 @ np.hstack([c, np.ones((8, 1))]).T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    box = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
    assert np.allclose(box, (718.08, 135.72, 830.21, 302.70), atol=0.05)


def test_alpha_is_rotation_minus_viewing_angle():
    """alpha（观察角）= rotation_y − atan2(x, z)：0.01 − atan2(1.84, 8.41) ≈ −0.205，标注里写的是 −0.20。"""
    d = parse_kitti_label(LABEL_000000)
    assert math.isclose(d["rotation_y"] - math.atan2(d["x"], d["z"]), d["alpha"], abs_tol=0.01)


def test_nuscenes_global_to_camera():
    """车在世界 (100, 200, 0)、朝 +y（转了 90°）；另一辆车中心在世界 (100, 230, 1)，也就是我正前方 30 米、高 1 米。
    前视相机装在车上 (1.5, 0, 1.5)：相机坐标 = (0, 0.5, 28.5)（正前方、往下 0.5 米、深 28.5 米）。"""
    s = json.loads(Path("data/nuscenes_sample.json").read_text(encoding="utf-8"))
    p = global_to_camera(np.array([s["sample_annotation"]["translation"]]), s["ego_pose"], s["calibrated_sensor"])
    assert np.allclose(p, [[0.0, 0.5, 28.5]], atol=1e-5)
