# -*- coding: utf-8 -*-
"""第 5 章插图：I-JEPA 三网络结构、多块掩码、像素目标与表征目标的对比。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs


# ---------------------------------------------------------------- 小工具
def grid_cells(ax, x0, y0, w, h, n, cell_fc, edge="white", lw=0.7, alpha=1.0, z=3):
    """在 (x0,y0,w,h) 里画 n x n 个方格，cell_fc 可以是单色或函数 f(i,j)->color。"""
    cw, ch = w / n, h / n
    for i in range(n):
        for j in range(n):
            fc = cell_fc(i, j) if callable(cell_fc) else cell_fc
            ax.add_patch(Rectangle((x0 + j * cw, y0 + (n - 1 - i) * ch), cw, ch,
                                   facecolor=fc, edgecolor=edge, lw=lw,
                                   alpha=alpha, zorder=z))


def draw_values(ax, x0, y0, w, h, vals, edge="white", lw=0.55, alpha=1.0, z=3):
    """把一个 n x m 的灰度数组画成网格，vals 取值 0（黑）到 1（白）。"""
    n, m = vals.shape
    cw, ch = w / m, h / n
    for i in range(n):
        for j in range(m):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + (n - 1 - i) * ch), cw, ch,
                                   facecolor=str(float(vals[i, j])), edgecolor=edge,
                                   lw=lw, alpha=alpha, zorder=z))


# ---------------------------------------------------------------- 图 5-1
def fig_three_networks():
    fig = plt.figure(figsize=(7.0, 3.45))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 输入图像
    grid_cells(ax, 0.022, 0.36, 0.138, 0.28, 6, fs.LIGHT, edge="#FFFFFF")
    cw = 0.138 / 6; ch = 0.28 / 6
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 3)]:
        ax.add_patch(Rectangle((0.022 + j * cw, 0.36 + (6 - 1 - i) * ch), cw, ch,
                               facecolor="#D9C7B2", edgecolor="white", lw=0.7, zorder=4))
    fs.label(ax, 0.091, 0.325, "输入图像", fs=8.4, color=fs.MUTED)

    # 目标块 / 上下文块
    grid_cells(ax, 0.190, 0.68, 0.099, 0.20, 4, "#E7D8C6")
    cw = 0.099 / 4; ch = 0.20 / 4
    for (i, j) in [(1, 1), (1, 2), (2, 1)]:
        ax.add_patch(Rectangle((0.190 + j * cw, 0.68 + (4 - 1 - i) * ch), cw, ch,
                               facecolor="#D9C7B2", edgecolor="white", lw=0.7, zorder=4))
    fs.label(ax, 0.2395, 0.645, "目标块 $y$", fs=8.4, color=fs.MUTED)
    grid_cells(ax, 0.190, 0.16, 0.099, 0.20, 4, fs.LIGHT)
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        ax.add_patch(Rectangle((0.190 + j * cw, 0.16 + (4 - 1 - i) * ch), cw, ch,
                               facecolor="#FFFFFF", edgecolor="#D1D5DB", lw=0.7, zorder=4))
    fs.label(ax, 0.2395, 0.125, "上下文块 $x$", fs=8.4, color=fs.MUTED)

    # 编码器与预测器
    fs.box(ax, 0.320, 0.68, 0.115, 0.20, "目标编码器\n$f_{\\bar\\theta}$", fc="#F6F6F6",
           ec=fs.FAINT, fs=8.4, lw=1.1)
    fs.box(ax, 0.320, 0.16, 0.115, 0.20, "上下文编码器\n$f_\\theta$", fc="white",
           ec=fs.ACCENT, fs=8.4, lw=1.4)
    fs.box(ax, 0.462, 0.16, 0.110, 0.20, "预测器\n$g_\\phi$", fc="white",
           ec=fs.ACCENT, fs=8.4, lw=1.4)

    # 表征
    fs.box(ax, 0.462, 0.70, 0.100, 0.16, "目标表征\n$s_y$", fc="#F6F6F6",
           ec=fs.FAINT, fs=8.4, lw=1.1)
    fs.box(ax, 0.600, 0.185, 0.100, 0.15, "预测\n$\\hat{s}_y$", fc="#FFF7F0",
           ec=fs.ACCENT, fs=8.4, lw=1.2)

    # 位置 token
    fs.box(ax, 0.462, 0.408, 0.110, 0.056, "位置 token", fc="#FBF7F0",
           ec=fs.FAINT, fs=7.6, lw=0.9, rounded=0.014)

    # 箭头
    fs.arrow(ax, (0.162, 0.605), (0.187, 0.752), color=fs.MUTED, lw=1.0, rad=0.25)
    fs.arrow(ax, (0.162, 0.395), (0.187, 0.248), color=fs.MUTED, lw=1.0, rad=-0.25)
    fs.arrow(ax, (0.291, 0.780), (0.317, 0.780), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.437, 0.780), (0.459, 0.780), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.291, 0.260), (0.317, 0.260), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.437, 0.260), (0.459, 0.260), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.574, 0.260), (0.597, 0.260), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.517, 0.406), (0.517, 0.366), color=fs.MUTED, lw=1.0)

    # 损失
    fs.box(ax, 0.745, 0.400, 0.160, 0.200, "L2 损失\n$\\|\\hat{s}_y-s_y\\|^2$",
           fc="#FFFFFF", ec=fs.INK, fs=8.6, lw=1.2)
    fs.arrow(ax, (0.564, 0.780), (0.742, 0.585), color=fs.FAINT, lw=1.0,
             ls=(0, (4, 3)), rad=-0.14)
    fs.arrow(ax, (0.702, 0.260), (0.742, 0.412), color=fs.INK, lw=1.1, rad=-0.16)

    # 停梯度标记
    fs.label(ax, 0.611, 0.762, "$\\bot$", fs=11, color=fs.ACCENT,
             bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec=fs.ACCENT, lw=0.9))
    fs.label(ax, 0.545, 0.662, "不回传梯度", fs=7.4, color=fs.MUTED)

    # EMA 通路
    fs.arrow(ax, (0.3775, 0.365), (0.3775, 0.675), color=fs.PURPLE, lw=1.1,
             ls=(0, (3, 2)), ms=7)
    fs.label(ax, 0.394, 0.520, "EMA 复制权重", fs=7.6, color=fs.PURPLE, rotation=90)
    fs.label(ax, 0.424, 0.520, "0.996 $\\rightarrow$ 1.0", fs=7.4, color=fs.PURPLE, rotation=90)

    # 梯度回传通路：从损失下方折回上下文编码器
    ax.plot([0.825, 0.825], [0.398, 0.090], color=fs.ACCENT, lw=1.3,
            ls=(0, (5, 2.5)), zorder=2, solid_capstyle="butt")
    fs.arrow(ax, (0.825, 0.090), (0.398, 0.090), color=fs.ACCENT, lw=1.3,
             ls=(0, (5, 2.5)), ms=8)
    fs.label(ax, 0.610, 0.140, "梯度只沿这条路回传", fs=7.6, color=fs.ACCENT)

    # 图例
    ax.add_patch(Rectangle((0.290, 0.010), 0.022, 0.042, facecolor="white",
                           edgecolor=fs.ACCENT, lw=1.4, zorder=3))
    fs.label(ax, 0.320, 0.031, "参数由梯度训练", fs=7.6, color=fs.INK, ha="left")
    ax.add_patch(Rectangle((0.560, 0.010), 0.022, 0.042, facecolor="#F6F6F6",
                           edgecolor=fs.FAINT, lw=1.1, zorder=3))
    fs.label(ax, 0.590, 0.031, "权重由 EMA 复制", fs=7.6, color=fs.INK, ha="left")
    return fs.save(fig, "basics", "ch05", "three-networks")


# ---------------------------------------------------------------- 图 5-2
N = 16


def _photo():
    """16 x 16 的示意照片：一个壁龛里的陶罐，数值越大越亮。"""
    img = np.full((N, N), 0.95)
    for i in range(N):
        for j in range(N):
            d = np.hypot((i - 7.4) / 4.6, (j - 7.6) / 4.4)
            if d < 0.55:
                img[i, j] = 0.40
            elif d < 0.95:
                img[i, j] = 0.52
            elif d < 1.35:
                img[i, j] = 0.72
            elif d < 1.60:
                img[i, j] = 0.86
    # 一点纹理，避免看起来像纯色块
    rng = np.random.default_rng(11)
    img = np.clip(img + rng.normal(0, 0.022, img.shape), 0.0, 1.0)
    return img


BLOCKS = [  # (row_start, row_end_excl, col_start, col_end_excl, color)
    (1, 7, 1, 9, fs.ACCENT),
    (0, 7, 9, 15, fs.BLUE),
    (9, 15, 1, 9, fs.GREEN),
    (8, 15, 9, 15, fs.PURPLE),
]


def _cell_index():
    """返回每个 patch 属于哪个目标块（-1 表示不属于任何块）。"""
    idx = -np.ones((N, N), int)
    for k, (r0, r1, c0, c1, _) in enumerate(BLOCKS):
        idx[r0:r1, c0:c1] = k
    return idx


def fig_mask_blocks():
    idx = _cell_index()
    in_ctx = np.zeros((N, N), bool)
    in_ctx[0:15, 0:15] = True          # 上下文大块，占全图 88%
    kept = in_ctx & (idx < 0)
    img = _photo()

    fig = plt.figure(figsize=(6.9, 3.05))
    axes = [fig.add_axes([0.030, 0.185, 0.250, 0.640]),
            fig.add_axes([0.372, 0.185, 0.250, 0.640]),
            fig.add_axes([0.712, 0.185, 0.250, 0.640])]
    for a in axes:
        a.set_xlim(0, N); a.set_ylim(0, N); a.set_aspect("equal"); a.axis("off")

    # (a) 原图
    a0 = axes[0]
    draw_values(a0, 0, 0, N, N, img, lw=0.5)

    # (b) 四个目标块叠在原图上
    a1 = axes[1]
    draw_values(a1, 0, 0, N, N, img, lw=0.4)
    for i in range(N):
        for j in range(N):
            k = idx[i, j]
            if k >= 0:
                a1.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=BLOCKS[k][4],
                                       edgecolor="white", lw=0.4, alpha=0.34, zorder=4))
    for k, (r0, r1, c0, c1, col) in enumerate(BLOCKS):
        a1.add_patch(Rectangle((c0 + 0.06, N - r1 + 0.06), c1 - c0 - 0.12, r1 - r0 - 0.12,
                               facecolor="none", edgecolor="white", lw=2.4, zorder=5))
        a1.add_patch(Rectangle((c0 + 0.06, N - r1 + 0.06), c1 - c0 - 0.12, r1 - r0 - 0.12,
                               facecolor="none", edgecolor=col, lw=1.2, zorder=6))
        ncell = (r1 - r0) * (c1 - c0)
        a1.text((c0 + c1) / 2, N - (r0 + r1) / 2, f"{ncell / N / N * 100:.0f}%",
                ha="center", va="center", fontsize=7.0, color="white",
                weight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.20", fc=col, ec="white", lw=0.8))

    # (c) 上下文挖掉目标之后剩下的 patch
    a2 = axes[2]
    for i in range(N):
        for j in range(N):
            if kept[i, j]:
                fc = "#7FA6F0"
            elif in_ctx[i, j] and idx[i, j] >= 0:
                fc = "#F7EDE2"
            else:
                fc = "#FCFCFB"
            a2.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="#E2E2E0" if not in_ctx[i, j] else "white",
                                   lw=0.5, zorder=3))
    for k, (r0, r1, c0, c1, col) in enumerate(BLOCKS):
        a2.add_patch(Rectangle((c0 + 0.10, N - r1 + 0.10), c1 - c0 - 0.20, r1 - r0 - 0.20,
                               facecolor="none", edgecolor=col, lw=0.9,
                               ls=(0, (3, 2)), alpha=0.85, zorder=7))
    a2.text(4.2, 7.5, "可见", ha="center", va="center", fontsize=7.0, color="#1E3A8A",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#7FA6F0", lw=0.8,
                      alpha=0.95), zorder=8)
    a2.text(12.4, 3.0, "挖掉", ha="center", va="center", fontsize=7.0, color="#9A3412",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#E7CBB2", lw=0.8,
                      alpha=0.95), zorder=8)
    a2.text(N / 2, N + 0.5, "上下文大块 $15 \\times 15$，挖掉四块", ha="center", va="bottom",
            fontsize=7.4, color=fs.INK)

    caps = ["（a）原图，切成 16 $\\times$ 16 个 patch",
            "（b）四个目标块，各占一成五到两成",
            f"（c）挖去目标区，留下 {kept.sum() / N / N * 100:.0f}% 可见"]
    for x, cap in zip([0.155, 0.497, 0.837], caps):
        fig.text(x, 0.075, cap, ha="center", va="center", fontsize=8.2, color=fs.INK)

    print("target ratios:", [f"{(r1-r0)*(c1-c0)/N/N*100:.1f}%" for r0, r1, c0, c1, _ in BLOCKS],
          "context kept:", f"{kept.sum() / N / N * 100:.1f}%")
    return fs.save(fig, "basics", "ch05", "mask-blocks")


# ---------------------------------------------------------------- 图 5-3
def _blob(n=8, cx=3.5, cy=3.3, rx=2.3, ry=2.6, dx=0.0, dy=0.0, seed=0):
    """n x n 的示意图案，模拟一个物体的轮廓；dx/dy 让轮廓整体平移。"""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 0.93)
    for i in range(n):
        for j in range(n):
            d = np.hypot((i - cy - dy) / ry, (j - cx - dx) / rx)
            if d < 0.55:
                img[i, j] = 0.36
            elif d < 0.95:
                img[i, j] = 0.60
            elif d < 1.30:
                img[i, j] = 0.84
    return np.clip(img + rng.normal(0, 0.010, img.shape), 0, 1)


def _mean_field(n=8, k=7):
    """把 k 个位置略有偏移的合理答案平均起来，模拟平方误差下的条件均值。"""
    acc = np.zeros((n, n))
    for s in range(k):
        dx = 0.55 * np.cos(2 * np.pi * s / k)
        dy = 0.55 * np.sin(2 * np.pi * s / k)
        acc += _blob(n, dx=dx, dy=dy, seed=s)
    return acc / k


def fig_pixel_vs_latent():
    fig = plt.figure(figsize=(7.0, 3.7))
    ax = fig.add_axes([0.0, 0.0, 0.695, 1.0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    n = 8
    gw = 0.150
    gh = 0.200
    ctx = _blob(n, seed=99)

    # ---------- (a) 像素空间 ----------
    ax.text(0.020, 0.930, "（a）把靶子放在像素上", fontsize=8.8, color=fs.GOLD,
            weight="bold", ha="left", va="center")
    y_a = 0.640
    draw_values(ax, 0.020, y_a, gw, gh, ctx)
    fs.label(ax, 0.020 + gw / 2, y_a - 0.038, "上下文块", fs=7.8, color=fs.MUTED)

    fs.box(ax, 0.212, y_a + 0.012, 0.098, gh - 0.024, "编码器", fc="white", ec=fs.INK,
           fs=8.2, lw=1.0)
    fs.box(ax, 0.340, y_a + 0.012, 0.098, gh - 0.024, "解码器", fc="white", ec=fs.GOLD,
           fs=8.2, lw=1.2)

    draw_values(ax, 0.472, y_a, gw, gh, _mean_field(n))
    draw_values(ax, 0.700, y_a, gw, gh, _blob(n, seed=3))
    fs.label(ax, 0.472 + gw / 2, y_a - 0.038, "还原出的像素", fs=7.8, color=fs.GOLD)
    fs.label(ax, 0.472 + gw / 2, y_a - 0.080, "所有合理答案的平均", fs=7.2, color=fs.FAINT)
    fs.label(ax, 0.700 + gw / 2, y_a - 0.038, "原图像素", fs=7.8, color=fs.MUTED)

    ymid = y_a + gh / 2
    fs.arrow(ax, (0.174, ymid), (0.209, ymid), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.313, ymid), (0.337, ymid), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.441, ymid), (0.469, ymid), color=fs.MUTED, lw=1.0)
    ax.annotate("", xy=(0.695, ymid), xytext=(0.627, ymid),
                arrowprops=dict(arrowstyle="<->", color=fs.GOLD, lw=1.0))
    fs.label(ax, 0.661, y_a + gh + 0.048, "逐像素比", fs=7.4, color=fs.GOLD)

    # ---------- (b) 表征空间 ----------
    ax.text(0.020, 0.440, "（b）把靶子放在表征上", fontsize=8.8, color=fs.ACCENT,
            weight="bold", ha="left", va="center")
    y_b = 0.130
    draw_values(ax, 0.020, y_b, gw, gh, ctx)
    fs.label(ax, 0.020 + gw / 2, y_b - 0.038, "上下文块", fs=7.8, color=fs.MUTED)

    fs.box(ax, 0.212, y_b + 0.012, 0.098, gh - 0.024, "编码器", fc="white", ec=fs.INK,
           fs=8.2, lw=1.0)
    fs.box(ax, 0.340, y_b + 0.012, 0.098, gh - 0.024, "预测器", fc="white", ec=fs.ACCENT,
           fs=8.2, lw=1.2)
    ymid_b = y_b + gh / 2
    fs.arrow(ax, (0.174, ymid_b), (0.209, ymid_b), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.313, ymid_b), (0.337, ymid_b), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.441, ymid_b), (0.469, ymid_b), color=fs.MUTED, lw=1.0)

    # 两个向量：一列小格
    rng = np.random.default_rng(7)
    va = np.clip(rng.normal(0.52, 0.19, 8), 0.08, 0.94)
    vb = np.clip(va + rng.normal(0, 0.08, 8), 0.08, 0.94)
    bw = 0.030
    for k in range(8):
        for (x0, vv, col) in [(0.492, va[k], fs.LIGHT), (0.652, vb[k], fs.LIGHT)]:
            ax.add_patch(Rectangle((x0, y_b + 0.006 + k * (gh - 0.012) / 8), bw,
                                   (gh - 0.012) / 8, facecolor=str(float(vv)),
                                   edgecolor="white", lw=0.5, zorder=4))
    fs.label(ax, 0.492 + bw / 2, y_b - 0.038, "预测 $\\hat{s}_y$", fs=7.8, color=fs.ACCENT)
    fs.label(ax, 0.652 + bw / 2, y_b - 0.038, "目标 $s_y$", fs=7.8, color=fs.MUTED)
    ax.annotate("", xy=(0.646, ymid_b), xytext=(0.528, ymid_b),
                arrowprops=dict(arrowstyle="<->", color=fs.ACCENT, lw=1.0))
    fs.label(ax, 0.587, y_b + gh + 0.048, "只比一个向量", fs=7.4, color=fs.ACCENT)

    # ---------- (c) 柱状对比 ----------
    axb = fig.add_axes([0.760, 0.230, 0.220, 0.520])
    vals = [40.7, 66.9]
    axb.bar([0, 1], vals, color=[fs.FAINT, fs.ACCENT], width=0.55, zorder=3)
    for x, v in zip([0, 1], vals):
        axb.text(x, v + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=9.0,
                 color=fs.INK)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["像素目标", "表征目标"], fontsize=8.2)
    axb.set_xlim(-0.55, 1.55)
    axb.set_ylim(0, 84)
    axb.set_yticks([0, 20, 40, 60, 80])
    axb.set_ylabel("1% 标签线性探测 top-1", fontsize=8.0, labelpad=3)
    axb.tick_params(labelsize=7.4)
    fs.clean(axb, spines=("left", "bottom"))
    axb.set_title("（c）两种靶子的下游精度", fontsize=8.2, color=fs.INK, pad=8)
    fig.text(0.870, 0.078, "ViT-L/16，同一套掩码", ha="center", va="center",
             fontsize=7.4, color=fs.MUTED)
    return fs.save(fig, "basics", "ch05", "pixel-vs-latent")


# ---------------------------------------------------------------- 图 5-4
def _thumb(ax, x0, y0, w, n=4, on="#D9C7B2"):
    """一小块方形缩略图，表示输入图；n x n 个格子。"""
    h = w * 9.0 / 4.25
    grid_cells(ax, x0, y0, w, h, n, fs.LIGHT, edge="#FFFFFF", lw=0.5)
    cw, ch = w / n, h / n
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]:
        ax.add_patch(Rectangle((x0 + j * cw, y0 + (n - 1 - i) * ch), cw, ch,
                               facecolor=on, edgecolor="white", lw=0.5, zorder=4))
    return h


def fig_info_flow():
    fig = plt.figure(figsize=(9.0, 4.25))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ============ (a) MAE ============
    ax.text(0.030, 0.945, "（a）MAE：梯度要穿过解码器", fontsize=9.2, color=fs.GOLD,
            weight="bold", ha="left", va="center")
    h = _thumb(ax, 0.030, 0.715, 0.045)
    fs.label(ax, 0.0525, 0.690, "输入图", fs=8.0, color=fs.MUTED)
    fs.box(ax, 0.105, 0.690, 0.085, 0.135, "编码器\n632M", fc="white", ec=fs.ACCENT,
           fs=8.2, lw=1.3)
    fs.box(ax, 0.245, 0.690, 0.085, 0.135, "解码器\n512 维", fc="white", ec=fs.GOLD,
           fs=8.2, lw=1.3)
    fs.box(ax, 0.385, 0.690, 0.075, 0.135, "重建\n像素", fc="white", ec=fs.INK,
           fs=8.2, lw=1.0)
    fs.arrow(ax, (0.077, 0.7575), (0.103, 0.7575), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.192, 0.7575), (0.243, 0.7575), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.332, 0.7575), (0.383, 0.7575), color=fs.MUTED, lw=1.0)

    fs.box(ax, 0.375, 0.500, 0.090, 0.075, "像素损失", fc="#FFFFFF", ec=fs.INK,
           fs=8.2, lw=1.0)
    fs.arrow(ax, (0.4225, 0.688), (0.4225, 0.577), color=fs.MUTED, lw=1.0)

    # 梯度：损失 → 解码器 → 编码器
    ax.plot([0.373, 0.1475], [0.535, 0.535], color=fs.ACCENT, lw=1.3,
            ls=(0, (5, 2.5)), zorder=2, solid_capstyle="butt")
    fs.arrow(ax, (0.1475, 0.535), (0.1475, 0.687), color=fs.ACCENT, lw=1.3,
             ls=(0, (5, 2.5)), ms=8)
    fs.label(ax, 0.262, 0.487, "梯度原路返回，穿过解码器", fs=8.0, color=fs.ACCENT)

    for i, line in enumerate([
            "编码器只看 64 个可见 token，占全图 25%。",
            "解码器处理全部 256 个 token，8 层 512 维，",
            "约 25M 参数（按宽度深度估算），每 token 的",
            "FLOPs 约为 ViT-L 的 9%。预训练后解码器丢弃。"]):
        fs.label(ax, 0.030, 0.400 - i * 0.056, line, fs=8.0, color=fs.INK, ha="left")

    # ============ (b) I-JEPA ============
    ax.text(0.510, 0.945, "（b）I-JEPA：梯度只到上下文编码器", fontsize=9.2,
            color=fs.ACCENT, weight="bold", ha="left", va="center")
    _thumb(ax, 0.510, 0.508, 0.045)
    fs.label(ax, 0.5325, 0.483, "输入图", fs=8.0, color=fs.MUTED)

    fs.box(ax, 0.585, 0.748, 0.095, 0.122, "目标编码器\n632M", fc="#F6F6F6",
           ec=fs.FAINT, fs=8.2, lw=1.0)
    fs.box(ax, 0.735, 0.748, 0.085, 0.122, "目标向量", fc="#F6F6F6", ec=fs.FAINT,
           fs=8.2, lw=1.0)
    fs.box(ax, 0.875, 0.748, 0.085, 0.122, "L2 损失", fc="white", ec=fs.INK,
           fs=8.2, lw=1.2)
    fs.box(ax, 0.585, 0.448, 0.095, 0.122, "上下文\n编码器", fc="white", ec=fs.ACCENT,
           fs=8.2, lw=1.3)
    fs.box(ax, 0.735, 0.448, 0.095, 0.122, "预测器\n约 22M", fc="white", ec=fs.ACCENT,
           fs=8.2, lw=1.3)
    fs.box(ax, 0.875, 0.448, 0.085, 0.122, "预测\n向量", fc="white", ec=fs.ACCENT,
           fs=8.2, lw=1.3)
    fs.box(ax, 0.722, 0.596, 0.120, 0.056, "位置 token", fc="#FBF7F0", ec=fs.FAINT,
           fs=8.0, lw=1.0, rounded=0.014)

    fs.arrow(ax, (0.558, 0.588), (0.583, 0.740), color=fs.MUTED, lw=1.0, rad=-0.25)
    fs.arrow(ax, (0.558, 0.518), (0.583, 0.512), color=fs.MUTED, lw=1.0, rad=0.2)
    fs.arrow(ax, (0.682, 0.809), (0.733, 0.809), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.822, 0.809), (0.873, 0.809), color=fs.MUTED, lw=1.0)

    # 前向（实线）与梯度（虚线）并排
    fs.arrow(ax, (0.682, 0.524), (0.733, 0.524), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.733, 0.498), (0.684, 0.498), color=fs.ACCENT, lw=1.2,
             ls=(0, (5, 2.5)))
    fs.arrow(ax, (0.832, 0.524), (0.873, 0.524), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.873, 0.498), (0.834, 0.498), color=fs.ACCENT, lw=1.2,
             ls=(0, (5, 2.5)))
    fs.arrow(ax, (0.930, 0.572), (0.930, 0.745), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.902, 0.745), (0.902, 0.580), color=fs.ACCENT, lw=1.2,
             ls=(0, (5, 2.5)))
    fs.arrow(ax, (0.7825, 0.596), (0.7825, 0.572), color=fs.MUTED, lw=0.9, ms=7)

    fs.label(ax, 0.7075, 0.809, "$\\bot$", fs=11, color=fs.ACCENT,
             bbox=dict(boxstyle="circle,pad=0.14", fc="white", ec=fs.ACCENT, lw=0.9))
    fs.label(ax, 0.7075, 0.700, "停梯度", fs=8.0, color=fs.ACCENT)
    fs.label(ax, 0.916, 0.662, "梯度", fs=8.0, color=fs.ACCENT, rotation=90)

    for i, line in enumerate([
            "目标编码器整图前向一次，256 个 token，不记梯度；",
            "多这一次前向让单步慢约 7%，收敛只要五分之一",
            "左右的迭代。预测器 384 维 12 层，跑四遍。",
            "上下文编码器只处理可见 token，反向到它为止。"]):
        fs.label(ax, 0.510, 0.400 - i * 0.056, line, fs=8.0, color=fs.INK, ha="left")

    fig.text(0.5, 0.028, "实线示前向，橙色虚线示梯度方向，$\\bot$ 示停梯度",
             ha="center", va="center", fontsize=8.4, color=fs.MUTED)
    return fs.save(fig, "basics", "ch05", "info-flow")


# ---------------------------------------------------------------- 图 5-5
def _union_area(rects):
    """若干轴对齐矩形的并集面积，矩形按 (x0, y0, x1, y1) 给，用容斥逐项算。"""
    n = len(rects)
    total = 0.0
    for mask in range(1, 1 << n):
        x0, y0, x1, y1 = -1e9, -1e9, 1e9, 1e9
        bits = 0
        for k in range(n):
            if mask >> k & 1:
                bits += 1
                x0 = max(x0, rects[k][0]); y0 = max(y0, rects[k][1])
                x1 = min(x1, rects[k][2]); y1 = min(y1, rects[k][3])
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        total += area if bits % 2 else -area
    return total


def _sim_masks(n=3000, seed=20261127):
    """按论文的比例区间采样上下文块与四个目标块，返回可见比例与泄漏比例。
    上下文块面积 0.85–1.0、长宽比为 1；目标块面积 0.15–0.2、长宽比 0.75–1.5，
    四个块各落在一个象限附近，避免互相盖住。"""
    rng = np.random.default_rng(seed)
    vis_rm, vis_raw, leak = [], [], []
    for _ in range(n):
        side = float(rng.uniform(0.85, 1.0)) ** 0.5
        cx = rng.uniform(side / 2, 1 - side / 2)
        cy = rng.uniform(side / 2, 1 - side / 2)
        ctx = (cx - side / 2, cy - side / 2, side, side)
        boxes = []
        for k in range(4):
            cxx = 0.25 + 0.5 * (k % 2) + rng.uniform(-0.05, 0.05)
            cyy = 0.25 + 0.5 * (k // 2) + rng.uniform(-0.05, 0.05)
            st = rng.uniform(0.15, 0.2)
            ar = rng.uniform(0.75, 1.5)
            w, hh = (st * ar) ** 0.5, (st / ar) ** 0.5
            x = float(np.clip(cxx, w / 2, 1 - w / 2))
            y = float(np.clip(cyy, hh / 2, 1 - hh / 2))
            boxes.append((x - w / 2, y - hh / 2, w, hh))
        clipped = [(max(b[0], ctx[0]), max(b[1], ctx[1]),
                    min(b[0] + b[2], ctx[0] + ctx[2]),
                    min(b[1] + b[3], ctx[1] + ctx[3])) for b in boxes]
        cov = _union_area(clipped)
        tgt = _union_area([(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in boxes])
        vis_rm.append(side * side - cov)
        vis_raw.append(side * side)
        leak.append(cov / tgt)
    return np.array(vis_rm), np.array(vis_raw), np.array(leak)


def fig_mask_overlap():
    idx = _cell_index()
    img = _photo()
    in_ctx = np.zeros((N, N), bool)
    in_ctx[0:15, 0:15] = True

    fig = plt.figure(figsize=(7.6, 3.55))
    axes = [fig.add_axes([0.030, 0.215, 0.215, 0.620]),
            fig.add_axes([0.283, 0.215, 0.215, 0.620])]
    for a in axes:
        a.set_xlim(0, N); a.set_ylim(0, 17.4); a.set_aspect("equal"); a.axis("off")

    for a, drop in zip(axes, [True, False]):
        for i in range(N):
            for j in range(N):
                k = idx[i, j]
                if not in_ctx[i, j]:
                    fc = "#F1F0ED"
                elif k >= 0 and drop:
                    fc = "#D6D6D2"          # 被挖掉的目标区
                elif k >= 0:
                    fc = "#F0A98E"          # 泄漏：目标区仍可见
                else:
                    fc = "#7FA6F0"          # 可见上下文
                a.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=fc,
                                       edgecolor="#E2E2E0", lw=0.45, zorder=3))
        for k, (r0, r1, c0, c1, col) in enumerate(BLOCKS):
            a.add_patch(Rectangle((c0 + 0.10, N - r1 + 0.10), c1 - c0 - 0.20, r1 - r0 - 0.20,
                                   facecolor="none", edgecolor=col, lw=1.0, zorder=7))
        a.add_patch(Rectangle((0.04, 1.04), 14.92, 14.92, facecolor="none",
                              edgecolor=fs.INK, lw=1.2, ls=(0, (4, 3)), zorder=8))

    kept = int((in_ctx & (idx < 0)).sum())
    tot = N * N
    axes[0].text(7.5, 16.2, f"可见 {kept} 格，占 {kept / tot * 100:.0f}%", ha="center",
                 va="bottom", fontsize=8.2, color=fs.INK)
    axes[1].text(7.5, 16.2, f"可见 {int(in_ctx.sum())} 格，占 {int(in_ctx.sum()) / tot * 100:.0f}%",
                 ha="center", va="bottom", fontsize=8.2, color="#9A3412")
    fig.text(0.137, 0.135, "（a）挖掉与目标重叠的部分", ha="center", va="center",
             fontsize=8.4, color=fs.INK)
    fig.text(0.390, 0.135, "（b）忘记挖，目标整块留在上下文里", ha="center", va="center",
             fontsize=8.4, color="#9A3412")
    fig.text(0.137, 0.075, f"蓝色可见，灰色是被挖掉的目标区", ha="center", va="center",
             fontsize=8.0, color=fs.MUTED)
    fig.text(0.390, 0.075, "橙色全是泄漏，预测器可以直接抄", ha="center", va="center",
             fontsize=8.0, color=fs.MUTED)

    # (c) 随机采样下两种做法的可见比例
    vis_rm, vis_raw, leak = _sim_masks()
    axc = fig.add_axes([0.600, 0.230, 0.360, 0.615])
    bins = np.linspace(0, 1, 41)
    n1, _, _ = axc.hist(vis_rm, bins=bins, color=fs.BLUE, alpha=0.72, label="挖掉重叠")
    n2, _, _ = axc.hist(vis_raw, bins=bins, color="#E8B4A2", alpha=0.85, label="不挖")
    ymax = max(n1.max(), n2.max())
    for v, c in [(vis_rm.mean(), fs.BLUE), (vis_raw.mean(), "#9A3412")]:
        axc.axvline(v, color=c, lw=1.1, ls=(0, (4, 2.5)), zorder=4)
    axc.set_xlim(0, 1)
    axc.set_ylim(0, ymax * 1.55)
    axc.set_yticks([])
    axc.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axc.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8.0)
    axc.set_xlabel("送进上下文编码器的 patch 比例", fontsize=8.0, labelpad=3)
    fs.clean(axc, spines=("bottom",))
    axc.set_title("（c）三千次随机掩码的可见比例", fontsize=8.4, color=fs.INK, pad=7)
    axc.annotate(f"均值 {vis_rm.mean() * 100:.0f}%", xy=(vis_rm.mean(), ymax * 0.72),
                 xytext=(0.42, ymax * 1.08), fontsize=8.2, color=fs.BLUE,
                 arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=0.9))
    axc.annotate(f"均值 {vis_raw.mean() * 100:.0f}%", xy=(vis_raw.mean(), ymax * 0.52),
                 xytext=(0.55, ymax * 0.80), fontsize=8.2, color="#9A3412",
                 arrowprops=dict(arrowstyle="-|>", color="#9A3412", lw=0.9))
    axc.legend(loc="upper center", fontsize=8.0, handlelength=1.1, ncol=2,
               bbox_to_anchor=(0.52, 1.04))
    fig.text(0.780, 0.075,
             f"目标块面积有 {np.median(leak) * 100:.0f}% 落在上下文里（中位数）",
             ha="center", va="center", fontsize=8.0, color=fs.MUTED)
    print("visible removed mean %.3f median %.3f | raw %.3f | leak median %.3f min %.3f | frac>0.9 %.3f"
          % (vis_rm.mean(), np.median(vis_rm), vis_raw.mean(), np.median(leak), leak.min(),
             float((leak > 0.9).mean())))
    return fs.save(fig, "basics", "ch05", "mask-overlap")


if __name__ == "__main__":
    fig_three_networks()
    fig_mask_blocks()
    fig_pixel_vs_latent()
    fig_info_flow()
    fig_mask_overlap()
