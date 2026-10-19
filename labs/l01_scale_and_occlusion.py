"""L01 实验：同一辆车放在 10 / 20 / 40 米，照片里和俯视图里各多大？卡车挡在前面时呢？

这个实验要回答：BEV 课件说俯视图“尺度变化小、遮挡小”，具体小多少？

运行（在仓库根目录）：python -m labs.l01_scale_and_occlusion
"""

from pathlib import Path

from bev.geometry import ego_to_bev_pixel, project_to_image

F_PX, CX, CY = 1000.0, 800.0, 450.0   # 照片 1600 × 900（nuScenes 的分辨率；它的真实焦距约 1266 像素，这里取 1000 好手算）
CAM_HEIGHT_M = 1.5                   # 相机装在离地 1.5 米
RES_M, BEV_PX = 0.512, 200           # 俯视图：200 × 200 格，每格 0.512 米

CAR = {"name": "轿车", "length": 4.7, "width": 1.9, "height": 1.5}
TRUCK = {"name": "卡车", "length": 8.0, "width": 2.5, "height": 3.5}


def ego_to_cam(x_e, y_e, z_e):
    """车坐标（x 前 y 左 z 上）→ 相机坐标（x 右 y 下 z 前），相机朝正前方、装在 CAM_HEIGHT_M 高。L02 会用旋转矩阵正式写这一步。"""
    return -y_e, CAM_HEIGHT_M - z_e, x_e


def front_face_in_image(veh, dist_m):
    """车尾那一面（离我们最近的一面）在照片里的框：返回 (u_left, v_top, u_right, v_bottom)。"""
    w, h = veh["width"], veh["height"]
    u_l, v_b = project_to_image(*ego_to_cam(dist_m, w / 2, 0.0), F_PX, CX, CY)     # 左下角（y 左为正）
    u_r, v_t = project_to_image(*ego_to_cam(dist_m, -w / 2, h), F_PX, CX, CY)      # 右上角
    return u_l, v_t, u_r, v_b


def footprint_in_bev(veh, dist_m):
    """车在俯视图里占的格子：返回 (row_front, col_left, row_back, col_right)。dist_m 是车尾离我的距离。"""
    l, w = veh["length"], veh["width"]
    r_front, c_left = ego_to_bev_pixel(dist_m + l, w / 2, RES_M, BEV_PX)
    r_back, c_right = ego_to_bev_pixel(dist_m, -w / 2, RES_M, BEV_PX)
    return r_front, c_left, r_back, c_right


def main():
    print("同一辆轿车（高 1.5 米、长 4.7 米）放在不同距离：")
    print(f"{'距离':>6} | {'照片里高（像素）':>10} | {'俯视图里长（格）':>10}")
    for d in (10.0, 20.0, 40.0):
        u_l, v_t, u_r, v_b = front_face_in_image(CAR, d)
        r_f, _, r_b, _ = footprint_in_bev(CAR, d)
        print(f"{d:5.0f} 米 | {v_b - v_t:14.1f} | {r_b - r_f:14.2f}")

    print("\n卡车（高 3.5 米）在 15 米，轿车在 30 米，同一车道：")
    t = front_face_in_image(TRUCK, 15.0)
    c = front_face_in_image(CAR, 30.0)
    hidden = t[0] <= c[0] and c[2] <= t[2] and t[1] <= c[1] and c[3] <= t[3]
    print(f"  照片里：卡车占 u {t[0]:.1f}–{t[2]:.1f}、v {t[1]:.1f}–{t[3]:.1f}；轿车占 u {c[0]:.1f}–{c[2]:.1f}、v {c[1]:.1f}–{c[3]:.1f}"
          f" → 轿车{'被卡车完全挡住' if hidden else '露出来了'}")
    bt, bc = footprint_in_bev(TRUCK, 15.0), footprint_in_bev(CAR, 30.0)
    print(f"  俯视图里：卡车占第 {bt[0]:.1f}–{bt[2]:.1f} 行，轿车占第 {bc[0]:.1f}–{bc[2]:.1f} 行 → 各在各的格子里，都看得见")
    plot()


# ---------- 以下是画图，助手写好的，不用手敲 ----------

def plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("没有装 matplotlib，跳过画图。需要的话运行：python -m pip install matplotlib")
        return
    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax_img.set_xlim(0, 1600); ax_img.set_ylim(900, 0); ax_img.set_aspect("equal")
    ax_img.axhline(CY, color="#9a9a94", linewidth=1, linestyle=":")
    ax_img.text(10, CY - 8, "horizon", color="#9a9a94", fontsize=8)
    colors = {10.0: "#1e66d6", 20.0: "#1baf7a", 40.0: "#eb6834"}
    for d, col in colors.items():
        u_l, v_t, u_r, v_b = front_face_in_image(CAR, d)
        ax_img.add_patch(Rectangle((u_l, v_t), u_r - u_l, v_b - v_t, fill=False, edgecolor=col, linewidth=2))
        ax_img.text(u_r + 6, v_b, f"{d:.0f} m: {v_b - v_t:.1f} px", color=col, fontsize=9)
    ax_img.set_title("front camera image (1600 x 900): same car at 10 / 20 / 40 m")
    ax_bev.set_xlim(60, 140); ax_bev.set_ylim(110, 0); ax_bev.set_aspect("equal")
    ax_bev.add_patch(Rectangle((100 - 0.95 / RES_M, 100 - 2.35 / RES_M), 1.9 / RES_M, 4.7 / RES_M, color="#52514e"))
    ax_bev.text(103, 102, "ego", fontsize=8)
    for d, col in colors.items():
        r_f, c_l, r_b, c_r = footprint_in_bev(CAR, d)
        ax_bev.add_patch(Rectangle((c_l, r_f), c_r - c_l, r_b - r_f, fill=False, edgecolor=col, linewidth=2))
        ax_bev.text(c_r + 1.5, (r_f + r_b) / 2, f"{d:.0f} m: {r_b - r_f:.2f} cells", color=col, fontsize=9, va="center")
    ax_bev.set_title("BEV grid (0.512 m / cell): same car, same size")
    for ax in (ax_img, ax_bev):
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = Path("outputs"); out.mkdir(exist_ok=True)
    fig.savefig(out / "l01_perspective_vs_bev.png", dpi=110)
    print("\n图已保存：outputs/l01_perspective_vs_bev.png")


if __name__ == "__main__":
    main()
