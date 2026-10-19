"""L02 测试：数字都能手算（f = 1000，照片 1600 × 900，光心 (800, 450)）。"""

import math

import numpy as np

from bev.camera import pixel_to_ray, project_points, transform_points
from bev.rig import K_DEFAULT, camera_extrinsics, make_T, rot_z

W, H = 1600, 900


def test_transform_translation_only():
    """只平移 (10, 0, 0)：点 (1, 2, 3) 变成 (11, 2, 3)。"""
    T = make_T(np.eye(3), [10.0, 0.0, 0.0])
    assert np.allclose(transform_points(T, np.array([[1.0, 2.0, 3.0]])), [[11.0, 2.0, 3.0]])


def test_transform_rotate_left_90():
    """绕 z 往左转 90°：车前方的点 (1, 0, 0) 跑到左边 (0, 1, 0)。"""
    T = make_T(rot_z(math.pi / 2), [0.0, 0.0, 0.0])
    assert np.allclose(transform_points(T, np.array([[1.0, 0.0, 0.0]])), [[0.0, 1.0, 0.0]])


def test_front_camera_center_and_left():
    """前视相机装在车中心 1.5 米高、朝前：正前方 10 米、1.5 米高的点落在 (800, 450)；往左 2 米的点 u = 800 − 1000·2/10 = 600。"""
    T = camera_extrinsics(0.0, 0.0, 1.5, 0.0)
    uv, depth, valid = project_points(K_DEFAULT, T, np.array([[10.0, 0.0, 1.5], [10.0, 2.0, 1.5]]), W, H)
    assert np.allclose(uv, [[800.0, 450.0], [600.0, 450.0]])
    assert np.allclose(depth, [10.0, 10.0]) and valid.all()


def test_behind_and_outside_are_invalid():
    """正后方 10 米的点：前视相机拍不到（depth = −10）；左边 20 米、前方 10 米：u = 800 − 2000 = −1200，出了照片。"""
    T = camera_extrinsics(0.0, 0.0, 1.5, 0.0)
    uv, depth, valid = project_points(K_DEFAULT, T, np.array([[-10.0, 0.0, 1.5], [10.0, 20.0, 1.5]]), W, H)
    assert depth[0] < 0 and not valid[0]
    assert math.isclose(uv[1, 0], -1200.0) and not valid[1]


def test_back_camera_sees_the_point_behind():
    """后视相机（朝 180°）：正后方 10 米的点落在照片正中间。"""
    T = camera_extrinsics(0.0, 0.0, 1.5, 180.0)
    uv, depth, valid = project_points(K_DEFAULT, T, np.array([[-10.0, 0.0, 1.5]]), W, H)
    assert np.allclose(uv, [[800.0, 450.0]]) and math.isclose(depth[0], 10.0) and valid[0]


def test_pixel_to_ray_and_round_trip():
    """像素 (800, 450) 的射线是 (0, 0, 1)；(900, 450) 是 (0.1, 0, 1)。射线 × 深度 10 再投回去，还是 (900, 450)。"""
    assert np.allclose(pixel_to_ray(K_DEFAULT, 800, 450), [0.0, 0.0, 1.0])
    ray = pixel_to_ray(K_DEFAULT, 900, 450)
    assert np.allclose(ray, [0.1, 0.0, 1.0])
    p_cam = ray * 10.0
    uvw = K_DEFAULT @ p_cam
    assert np.allclose(uvw[:2] / uvw[2], [900.0, 450.0])
