# -*- coding: utf-8 -*-
"""第 2 章插图：图像到向量、线性探测与微调、余弦与欧氏近邻。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, FancyBboxPatch
import figstyle as fs

# ---------- 文中引用的数字，全部在此算出 ----------
IMG_SIZE = 224
PATCH = 16
N_PATCH = (IMG_SIZE // PATCH) ** 2          # 224/16=14，14x14
HIDDEN = 768                                # ViT-Base / DINOv2 ViT-B/14 的嵌入维度
N_CLASS = 1000                              # ImageNet-1K
VIT_PARAMS = 86e6                           # ViT-Base 总参数（ViT 论文表 1）


def report_numbers():
    rng = np.random.default_rng(0)
    probe_params = HIDDEN * N_CLASS
    frac = probe_params / VIT_PARAMS

    # 高维随机向量两两距离的相对波动
    d = HIDDEN
    x = rng.normal(size=(1500, d))
    y = rng.normal(size=(1500, d))
    dist = np.linalg.norm(x - y, axis=1)
    rel = dist.std() / dist.mean()

    print(f"每边块数 {IMG_SIZE // PATCH}，块总数 {N_PATCH}")
    print(f"原图像素数 {IMG_SIZE * IMG_SIZE}，三通道数值个数 {IMG_SIZE * IMG_SIZE * 3}")
    print(f"线性探测需训练参数 {probe_params}，占 ViT-Base {VIT_PARAMS:.0f} 的 {frac*100:.2f}%")
    print(f"{d} 维随机两点距离 均值 {dist.mean():.2f} 标准差 {dist.std():.2f} 相对标准差 {rel*100:.2f}%")
    return dict(n_patch=N_PATCH, probe_frac=frac, rel_std=rel)


# ---------- 图 2-1 ----------
def fig_image_to_vector():
    fig = plt.figure(figsize=(6.7, 2.95))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # (a) 一张被切成小块的图
    cols = ["#BFD8EC", "#BFD8EC", "#F2D08A", "#BFD8EC",
            "#BFD8EC", "#E8C98F", "#BFD8EC", "#BFD8EC",
            "#7FA88A", "#6E9B7C", "#6E9B7C", "#8FB39A",
            "#6E9B7C", "#8B7A63", "#6E9B7C", "#6E9B7C"]
    cw, ch, n = 0.0415, 0.0415 * 7.0 / 3.05, 4
    x0, y0 = 0.026, 0.235
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + (n - 1 - i) * ch), cw, ch,
                                   facecolor=cols[i * n + j], edgecolor="white",
                                   linewidth=1.0, zorder=2))
    ax.add_patch(Rectangle((x0, y0), n * cw, n * ch, facecolor="none",
                           edgecolor=fs.FAINT, linewidth=0.8, zorder=3))
    fs.label(ax, x0 + n * cw / 2, 0.745, "一张图", fs=9.2)
    fs.label(ax, x0 + n * cw / 2, 0.135, "切成小块", fs=8.4, color=fs.MUTED)

    # 箭头
    fs.arrow(ax, (x0 + n * cw + 0.006, 0.455), (0.244, 0.455), color=fs.FAINT, lw=1.0, ms=6)

    # (b) 编码器
    fs.box(ax, 0.248, 0.335, 0.114, 0.275, "", fc="#F4F7FA", ec=fs.INK, lw=1.2)
    fs.label(ax, 0.305, 0.522, "编码器", fs=10, weight="bold")
    fs.label(ax, 0.305, 0.435, "$f_\\theta$", fs=10, color=fs.MUTED)
    fs.label(ax, 0.305, 0.255, "几千万个权重", fs=7.6, color=fs.MUTED)

    fs.arrow(ax, (0.366, 0.455), (0.400, 0.455), color=fs.INK, lw=1.1)

    # (c) 一个向量
    vx, vy, vw, vh = 0.404, 0.235, 0.043, 0.44
    vals = np.array([0.9, 0.2, -0.4, 0.7, 0.1, -0.8, 0.35, -0.15,
                     0.6, -0.55, 0.05, 0.8, -0.25, 0.45, -0.7, 0.3])
    k = len(vals)
    for i, v in enumerate(vals):
        c = plt.cm.RdBu_r((v + 1) / 2)
        ax.add_patch(Rectangle((vx, vy + i * vh / k), vw, vh / k,
                               facecolor=c, edgecolor="white", linewidth=0.6, zorder=2))
    ax.add_patch(Rectangle((vx, vy), vw, vh, facecolor="none", edgecolor=fs.FAINT,
                           linewidth=0.8, zorder=3))
    fs.label(ax, vx + vw / 2, 0.735, "768 个实数", fs=8.6)
    fs.label(ax, vx + vw / 2, 0.135, "一个向量", fs=8.4, color=fs.MUTED)

    fs.arrow(ax, (vx + vw + 0.006, 0.455), (0.487, 0.455), color=fs.INK, lw=1.1)

    # (d) 表征空间中的点
    ax3 = fig.add_axes([0.500, 0.045, 0.425, 0.87], projection="3d")
    rng = np.random.default_rng(7)
    centers = np.array([[-3.2, -1.6, 0.6], [2.6, -2.0, -1.4],
                        [0.4, 2.6, 1.2], [3.4, 1.8, 2.4]])
    for c, col in zip(centers, [fs.ACCENT, fs.BLUE, fs.GREEN, fs.PURPLE]):
        p = c + rng.normal(scale=0.72, size=(14, 3))
        ax3.scatter(p[:, 0], p[:, 1], p[:, 2], s=16, color=col, alpha=0.85,
                    edgecolor="white", linewidth=0.4, depthshade=False)
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    ax3.set_xlabel("维度 1", fontsize=7.6, labelpad=-8, color=fs.MUTED)
    ax3.set_ylabel("维度 2", fontsize=7.6, labelpad=-8, color=fs.MUTED)
    ax3.set_zlabel("维度 3", fontsize=7.6, labelpad=-10, color=fs.MUTED)
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#E5E7EB")
        pane.pane.set_alpha(1.0)
    ax3.grid(False)
    ax3.view_init(elev=16, azim=-58)
    return fs.save(fig, "basics", "ch02", "image-to-vector")


# ---------- 图 2-2 ----------
def fig_probe_vs_finetune():
    fig = plt.figure(figsize=(6.75, 2.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    panels = [
        dict(px=0.030, title="线性探测", sub="编码器不动，只学末尾一层",
             enc_fc="#F3F4F6", enc_note="冻结", grad="off",
             note="需要更新的参数约 0.77M，不到整个网络的 1%"),
        dict(px=0.520, title="端到端微调", sub="所有参数一起更新",
             enc_fc="#FDF1EA", enc_note="可训练", grad="on",
             note="86M 个参数全部跟着梯度走"),
    ]

    yc = 0.665
    for p in panels:
        px = p["px"]
        fs.label(ax, px + 0.215, 0.945, p["title"], fs=10.5, weight="bold")
        fs.label(ax, px + 0.215, 0.872, p["sub"], fs=8.2, color=fs.MUTED)

        fs.box(ax, px + 0.005, yc - 0.105, 0.072, 0.21, "图像", fc="#FFFFFF",
               ec=fs.FAINT, fs=8.8, lw=1.0)
        fs.arrow(ax, (px + 0.079, yc), (px + 0.099, yc), color=fs.FAINT, lw=1.0, ms=6)

        fs.box(ax, px + 0.102, yc - 0.125, 0.132, 0.25, "", fc=p["enc_fc"],
               ec=fs.ACCENT if p["grad"] == "on" else fs.FAINT, lw=1.2)
        if p["grad"] == "off":
            ax.add_patch(Rectangle((px + 0.102, yc - 0.125), 0.132, 0.25, facecolor="none",
                                   edgecolor=fs.FAINT, linewidth=1.0, linestyle=(0, (4, 2.6)),
                                   zorder=4))
        fs.label(ax, px + 0.168, yc + 0.045, "编码器", fs=9.2, weight="bold")
        fs.label(ax, px + 0.168, yc - 0.048, p["enc_note"], fs=8.2,
                 color=fs.MUTED if p["grad"] == "off" else fs.ACCENT)
        fs.arrow(ax, (px + 0.236, yc), (px + 0.256, yc), color=fs.FAINT, lw=1.0, ms=6)

        fs.box(ax, px + 0.259, yc - 0.095, 0.098, 0.19, "分类层", fc="#FFFFFF",
               ec=fs.ACCENT, fs=8.8, lw=1.3)
        fs.arrow(ax, (px + 0.359, yc), (px + 0.379, yc), color=fs.FAINT, lw=1.0, ms=6)
        fs.label(ax, px + 0.383, yc, "类别", fs=8.2, color=fs.MUTED, ha="left")

        y_lo = 0.455
        if p["grad"] == "off":
            ax.plot([px + 0.168, px + 0.168], [y_lo, yc - 0.135], color=fs.FAINT,
                    lw=0.9, linestyle=(0, (3, 2)), zorder=2)
            ax.plot([px + 0.155, px + 0.181], [y_lo - 0.014, y_lo + 0.014],
                    color=fs.FAINT, lw=1.0, zorder=5)
            ax.plot([px + 0.155, px + 0.181], [y_lo + 0.014, y_lo - 0.014],
                    color=fs.FAINT, lw=1.0, zorder=5)
            fs.label(ax, px + 0.196, y_lo, "梯度到这里为止", fs=7.8, color=fs.MUTED, ha="left")
        else:
            fs.arrow(ax, (px + 0.168, y_lo), (px + 0.168, yc - 0.135), color=fs.ACCENT,
                     lw=1.0, ms=9)
            fs.label(ax, px + 0.196, y_lo, "梯度一直传回第一层", fs=7.8,
                     color=fs.ACCENT, ha="left")

        fs.label(ax, px + 0.005, 0.325, p["note"], fs=8.0, color=fs.INK, ha="left")

    ax.plot([0.5, 0.5], [0.225, 0.99], color="#E5E7EB", lw=1.0)
    fs.label(ax, 0.5, 0.145, "左边量的是表征本身，右边量的是整个系统的上限", fs=8.4, color=fs.MUTED)
    return fs.save(fig, "basics", "ch02", "probe-vs-finetune")


# ---------- 图 2-3 ----------
def fig_metric_neighbors():
    q = np.array([2.20, 1.40])
    pts = {
        "A": np.array([4.57, 2.03]),
        "B": np.array([3.81, 2.20]),
        "C": np.array([2.68, 2.41]),
        "D": np.array([1.45, 2.51]),
        "E": np.array([2.69, 0.235]),
        "F": np.array([4.90, 4.90]),
        "G": np.array([0.60, 3.20]),
        "H": np.array([1.20, -1.25]),
        "I": np.array([5.80, 0.50]),
    }
    keys = list(pts)
    P = np.array([pts[k] for k in keys])

    d = np.linalg.norm(P - q, axis=1)
    cos = (P @ q) / (np.linalg.norm(P, axis=1) * np.linalg.norm(q))
    i_e = np.argsort(d)[:3]
    i_c = np.argsort(-cos)[:3]
    print("欧氏三近邻", [keys[i] for i in i_e], np.round(d[i_e], 2))
    print("余弦三近邻", [keys[i] for i in i_c], np.round(cos[i_c], 3))

    fig = plt.figure(figsize=(7.0, 3.62))
    axes = [fig.add_axes([0.055, 0.085, 0.395, 0.80]),
            fig.add_axes([0.545, 0.085, 0.395, 0.80])]
    titles = ["按欧氏距离取三个近邻", "按余弦相似度取三个近邻"]
    sel = [set(i_e), set(i_c)]

    for ax, title, chosen in zip(axes, titles, sel):
        ax.set_xlim(-0.4, 6.4); ax.set_ylim(-1.75, 5.35)
        ax.set_aspect("equal")
        fs.clean(ax, spines=("left", "bottom"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9.0, color=fs.INK, pad=6)

        for i, k in enumerate(keys):
            if i in chosen:
                ax.scatter(P[i, 0], P[i, 1], s=64, facecolor=fs.ACCENT,
                           edgecolor="white", linewidth=0.9, zorder=5)
            else:
                ax.scatter(P[i, 0], P[i, 1], s=26, facecolor="#CBD5E1",
                           edgecolor="white", linewidth=0.6, zorder=3)

        ax.scatter(q[0], q[1], s=150, marker="*", facecolor=fs.BLUE,
                   edgecolor="white", linewidth=0.8, zorder=6)
        ax.scatter([0], [0], s=12, color=fs.INK, zorder=4)
        fs.label(ax, 0.12, -0.42, "原点", fs=7.4, color=fs.MUTED, ha="left")
        fs.label(ax, q[0] + 0.10, q[1] - 0.55, "查询", fs=8.0, color=fs.BLUE, ha="left")

    # (a) 欧氏：一个圆
    ax = axes[0]
    r = d[i_e].max()
    ax.add_patch(plt.Circle(q, r, facecolor=fs.ACCENT, alpha=0.06,
                            edgecolor=fs.ACCENT, linewidth=0.9, linestyle=(0, (4, 2)),
                            zorder=2))
    ax.annotate("离得近，方向差得远", xy=tuple(pts["E"]), xytext=(3.35, -0.75),
                fontsize=7.8, color=fs.MUTED,
                arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.8,
                                connectionstyle="arc3,rad=0.15"))
    ax.annotate("方向几乎相同\n但距离很远", xy=tuple(pts["A"]), xytext=(0.35, 4.55),
                fontsize=7.8, color=fs.MUTED, ha="left",
                arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))

    # (b) 余弦：一个角度扇区
    ax = axes[1]
    a0 = np.degrees(np.arctan2(P[i_c, 1], P[i_c, 0])).min() - 1.0
    a1 = np.degrees(np.arctan2(P[i_c, 1], P[i_c, 0])).max() + 1.0
    ax.add_patch(Wedge((0, 0), 5.15, a0, a1, facecolor=fs.ACCENT, alpha=0.08,
                       edgecolor=fs.ACCENT, linewidth=0.9, linestyle=(0, (4, 2)),
                       zorder=2))
    for i in i_c:
        ax.plot([0, P[i, 0]], [0, P[i, 1]], color=fs.ACCENT, lw=0.6,
                linestyle=(0, (2, 2)), alpha=0.7, zorder=1)
    ax.annotate("离得近，但方向偏开\n落在扇区外", xy=tuple(pts["E"]), xytext=(3.35, -1.35),
                fontsize=7.8, color=fs.MUTED,
                arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.8,
                                connectionstyle="arc3,rad=-0.15"))

    return fs.save(fig, "basics", "ch02", "metric-neighbors")


if __name__ == "__main__":
    report_numbers()
    fig_image_to_vector()
    fig_probe_vs_finetune()
    fig_metric_neighbors()
