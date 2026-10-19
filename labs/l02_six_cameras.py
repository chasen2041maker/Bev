"""L02 实验：车上 6 个相机，周围 8 个物体，各被哪个相机拍到、落在哪个像素？

这个实验要回答：
  1. 环视 6 个相机怎么分工？有没有“两个相机都拍到”的重叠区、有没有谁都拍不到的盲区？
  2. 课件 009 第 10 页的投影公式省掉了光心 cx、cy，照着它算会差多少？

运行（在仓库根目录）：python -m labs.l02_six_cameras
"""

import math
from pathlib import Path

import numpy as np

from bev.camera import project_points
from bev.rig import HEIGHT, K_DEFAULT, RIG, WIDTH, make_K, rig_extrinsics

# 物体中心（车坐标系，z 取 0.75 米 = 轿车半高）
OBJECTS = {
    "A 正前 20 米": (20.0, 0.0),
    "B 左前 30°、15 米": (15.0 * math.cos(math.radians(30)), 15.0 * math.sin(math.radians(30))),
    "C 左前 63°": (5.0, 10.0),
    "D 左后": (-6.0, 8.0),
    "E 正后 15 米": (-15.0, 0.0),
    "F 右前": (10.0, -10.0),
    "G 正右 8 米": (0.0, -8.0),
    "H 车头前 2 米": (2.0, 0.0),
}


def main():
    extr = rig_extrinsics()
    pts = np.array([[x, y, 0.75] for x, y in OBJECTS.values()])
    seen_by = {name: [] for name in OBJECTS}
    for cam, T in extr.items():
        uv, depth, valid = project_points(K_DEFAULT, T, pts, WIDTH, HEIGHT)
        for i, name in enumerate(OBJECTS):
            if valid[i]:
                seen_by[name].append(f"{cam}({uv[i, 0]:.0f}, {uv[i, 1]:.0f})")
    print("每个物体被哪些相机拍到（括号里是像素 u, v）：")
    for name, cams in seen_by.items():
        tag = "  ← 重叠区，两个相机都拍到" if len(cams) >= 2 else ("  ← 盲区，谁都拍不到" if not cams else "")
        print(f"  {name:<14} {len(cams)} 个：{'；'.join(cams) if cams else '—'}{tag}")

    print("\n课件 009 第 10 页的公式没有 cx、cy（相当于光心在左上角）：")
    K_ppt = make_K(1000.0, 0.0, 0.0)
    T = extr["CAM_FRONT"]
    a = pts[:1]
    uv_ok, _, _ = project_points(K_DEFAULT, T, a, WIDTH, HEIGHT)
    uv_ppt, _, valid_ppt = project_points(K_ppt, T, a, WIDTH, HEIGHT)
    print(f"  A（正前 20 米）正确：({uv_ok[0, 0]:.1f}, {uv_ok[0, 1]:.1f})；照课件公式：({uv_ppt[0, 0]:.1f}, {uv_ppt[0, 1]:.1f})"
          f" → 偏到左上角，整整差了 (cx, cy) = (800, 450)")
    plot(extr, pts)


# ---------- 以下是画图，助手写好的，不用手敲 ----------
COLORS = ["#1e66d6", "#1baf7a", "#eb6834", "#7a4fd0", "#c4471f", "#5b6672"]


def plot(extr, pts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle
    except ImportError:
        print("没有装 matplotlib，跳过画图。需要的话运行：python -m pip install matplotlib")
        return
    fig = plt.figure(figsize=(15, 8))
    ax_bev = fig.add_axes([0.02, 0.05, 0.36, 0.9])
    half = math.atan(WIDTH / 2 / K_DEFAULT[0, 0])
    for (name, x, y, z, yaw), col in zip(RIG, COLORS):
        a0, a1 = math.radians(yaw) - half, math.radians(yaw) + half
        poly = [(y, x)] + [(y + 22 * math.sin(a), x + 22 * math.cos(a)) for a in np.linspace(a0, a1, 20)]
        ax_bev.add_patch(Polygon([(-py, px) for py, px in poly], closed=True, alpha=0.13, color=col))
        ax_bev.text(-(y + 23 * math.sin(math.radians(yaw))), x + 23 * math.cos(math.radians(yaw)), name.replace("CAM_", ""),
                    color=col, fontsize=8, ha="center", va="center")
    ax_bev.add_patch(Rectangle((-0.95, -1.0), 1.9, 4.7, color="#2b333c"))
    for (label, (x, y)) in zip(OBJECTS, [(p[0], p[1]) for p in pts]):
        ax_bev.plot(-y, x, "o", color="#d23b3b")
        ax_bev.text(-y + 0.6, x + 0.6, label.split()[0], fontsize=9)
    ax_bev.set_xlim(-25, 25); ax_bev.set_ylim(-25, 25); ax_bev.set_aspect("equal")
    ax_bev.set_title("top view (up = forward, left = left): 6 camera fields of view")
    ax_bev.set_xticks([]); ax_bev.set_yticks([])
    for k, ((name, *_), col) in enumerate(zip(RIG, COLORS)):
        ax = fig.add_axes([0.41 + (k % 3) * 0.196, 0.53 - (k // 3) * 0.45, 0.185, 0.38])
        ax.set_xlim(0, WIDTH); ax.set_ylim(HEIGHT, 0)
        ax.add_patch(Rectangle((0, 0), WIDTH, HEIGHT, fill=False, edgecolor=col, linewidth=2))
        uv, depth, valid = project_points(K_DEFAULT, extr[name], pts, WIDTH, HEIGHT)
        for label, (u, v), ok in zip(OBJECTS, uv, valid):
            if ok:
                ax.plot(u, v, "o", color="#d23b3b"); ax.text(u + 30, v - 30, label.split()[0], fontsize=9)
        ax.set_title(name, fontsize=9, color=col); ax.set_xticks([]); ax.set_yticks([])
    out = Path("outputs"); out.mkdir(exist_ok=True)
    fig.savefig(out / "l02_six_cameras.png", dpi=100)
    print("\n图已保存：outputs/l02_six_cameras.png")


if __name__ == "__main__":
    main()
