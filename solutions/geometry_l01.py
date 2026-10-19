"""L01 参考答案（先自己敲，卡住再看）。"""


def project_to_image(x_c, y_c, z_c, f_px, cx, cy):
    if z_c <= 0:
        raise ValueError("点在相机后面（z ≤ 0），拍不到")
    return f_px * x_c / z_c + cx, f_px * y_c / z_c + cy


def ego_to_bev_pixel(x_m, y_m, res_m=0.512, size_px=200):
    center = size_px / 2
    return center - x_m / res_m, center - y_m / res_m
