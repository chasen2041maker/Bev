"""L01 测试：数字都能手算（f = 1000 像素，照片 1600 × 900，光心 (800, 450)）。"""

import math

import pytest

from bev.geometry import ego_to_bev_pixel, project_to_image

F, CX, CY = 1000.0, 800.0, 450.0


def test_point_on_optical_axis_hits_center():
    """正前方 10 米、在光轴上的点：落在照片正中间 (800, 450)。"""
    assert project_to_image(0.0, 0.0, 10.0, F, CX, CY) == (800.0, 450.0)


def test_right_and_up_point():
    """前方 10 米、往右 1 米、往上 0.5 米（相机系 y 向下，所以 y = −0.5）：u = 1000·1/10 + 800 = 900，v = 1000·(−0.5)/10 + 450 = 400。"""
    u, v = project_to_image(1.0, -0.5, 10.0, F, CX, CY)
    assert math.isclose(u, 900.0) and math.isclose(v, 400.0)


def test_same_car_four_times_farther_is_four_times_smaller():
    """一辆 1.5 米高的车（相机装在 1.5 米高，车底在 y = 1.5、车顶在 y = 0）：10 米时 150 像素高，40 米时 37.5 像素高。"""
    def pixel_height(z):
        return project_to_image(0.0, 1.5, z, F, CX, CY)[1] - project_to_image(0.0, 0.0, z, F, CX, CY)[1]
    assert math.isclose(pixel_height(10.0), 150.0)
    assert math.isclose(pixel_height(40.0), 37.5)


def test_point_behind_camera_raises():
    """z ≤ 0 在相机后面，拍不到：抛 ValueError（后面 DETR3D 的“有效性掩码”就是干这个的）。"""
    with pytest.raises(ValueError):
        project_to_image(0.0, 0.0, -5.0, F, CX, CY)


def test_bev_center_and_directions():
    """200 × 200 格、每格 0.512 米：车在 (100, 100)；往前 10.24 米 → 行 80；往左 5.12 米 → 列 90；左后角 (−51.2, −51.2) → (200, 200)。"""
    assert ego_to_bev_pixel(0.0, 0.0) == (100.0, 100.0)
    row, col = ego_to_bev_pixel(10.24, 0.0)
    assert math.isclose(row, 80.0) and math.isclose(col, 100.0)
    row, col = ego_to_bev_pixel(0.0, 5.12)
    assert math.isclose(row, 100.0) and math.isclose(col, 90.0)
    row, col = ego_to_bev_pixel(-51.2, -51.2)
    assert math.isclose(row, 200.0) and math.isclose(col, 200.0)
