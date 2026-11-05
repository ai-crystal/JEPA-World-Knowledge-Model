# -*- coding: utf-8 -*-
"""第 8 章插图：I-JEPA 三网络结构、多块掩码、像素目标与表征目标的对比。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
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


def cell_font(ax, x0, y0, w, h, n, i, j, text, fs_=7.0, color=fs.INK, z=6):
    cw, ch = w / n, h / n
    ax.text(x0 + (j + 0.5) * cw, y0 + (n - 1 - i + 0.5) * ch, text,
            ha="center", va="center", fontsize=fs_, color=color, zorder=z)


# ---------------------------------------------------------------- 图 8-1
def fig_three_networks():
    fig = plt.figure(figsize=(7.0, 3.45))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 输入图像
    grid_cells(ax, 0.022, 0.36, 0.138, 0.28, 6, fs.LIGHT, edge="#FFFFFF")
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 3)]:
        cell_font(ax, 0.022, 0.36, 0.138, 0.28, 6, i, j, "", fs_=1)
        cw = 0.138 / 6; ch = 0.28 / 6
        ax.add_patch(Rectangle((0.022 + j * cw, 0.36 + (6 - 1 - i) * ch), cw, ch,
                               facecolor="#D9C7B2", edgecolor="white", lw=0.7, zorder=4))
    fs.label(ax, 0.091, 0.325, "输入图像", fs=8.4, color=fs.MUTED)

    # 目标块 / 上下文块
    grid_cells(ax, 0.190, 0.68, 0.099, 0.20, 4, "#E7D8C6")
    for (i, j) in [(1, 1), (1, 2), (2, 1)]:
        cw = 0.099 / 4; ch = 0.20 / 4
        ax.add_patch(Rectangle((0.190 + j * cw, 0.68 + (4 - 1 - i) * ch), cw, ch,
                               facecolor="#D9C7B2", edgecolor="white", lw=0.7, zorder=4))
    fs.label(ax, 0.2395, 0.645, "目标块 $y$", fs=8.4, color=fs.MUTED)
    grid_cells(ax, 0.190, 0.16, 0.099, 0.20, 4, fs.LIGHT)
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        cw = 0.099 / 4; ch = 0.20 / 4
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
    fs.label(ax, 0.609, 0.760, "$\\bot$", fs=11, color=fs.ACCENT,
             bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec=fs.ACCENT, lw=0.9))
    fs.label(ax, 0.580, 0.686, "不回传梯度", fs=7.4, color=fs.MUTED)

    # EMA 通路
    fs.arrow(ax, (0.3775, 0.365), (0.3775, 0.675), color=fs.PURPLE, lw=1.1,
             ls=(0, (3, 2)), ms=7)
    fs.label(ax, 0.394, 0.520, "EMA 复制权重", fs=7.6, color=fs.PURPLE, rotation=90)
    fs.label(ax, 0.424, 0.520, "0.996 $\\rightarrow$ 1.0", fs=7.4, color=fs.PURPLE, rotation=90)

    # 梯度回传通路
    fs.arrow(ax, (0.700, 0.072), (0.352, 0.072), color=fs.ACCENT, lw=1.3,
             ls=(0, (5, 2.5)), ms=8)
    fs.label(ax, 0.526, 0.120, "梯度只沿这条路回传", fs=7.6, color=fs.ACCENT)

    # 图例
    ax.add_patch(Rectangle((0.290, 0.012), 0.022, 0.042, facecolor="white",
                           edgecolor=fs.ACCENT, lw=1.4, zorder=3))
    fs.label(ax, 0.320, 0.033, "参数由梯度训练", fs=7.6, color=fs.INK, ha="left")
    ax.add_patch(Rectangle((0.560, 0.012), 0.022, 0.042, facecolor="#F6F6F6",
                           edgecolor=fs.FAINT, lw=1.1, zorder=3))
    fs.label(ax, 0.590, 0.033, "权重由 EMA 复制", fs=7.6, color=fs.INK, ha="left")
    return fs.save(fig, "ch08", "three-networks")


# ---------------------------------------------------------------- 图 8-2
BLOCKS = [  # (row_start, row_end_excl, col_start, col_end_excl, color)
    (0, 6, 0, 8, fs.ACCENT),
    (0, 7, 8, 15, fs.BLUE),
    (8, 15, 0, 6, fs.GREEN),
    (9, 15, 8, 15, fs.PURPLE),
]
N = 16


def _cell_index():
    """返回每个 patch 属于哪个目标块（-1 表示不属于任何块）。"""
    idx = -np.ones((N, N), int)
    for k, (r0, r1, c0, c1, _) in enumerate(BLOCKS):
        idx[r0:r1, c0:c1] = k
    return idx


def fig_mask_blocks():
    idx = _cell_index()
    ctx_r0, ctx_r1, ctx_c0, ctx_c1 = 0, 15, 0, 15          # 上下文大块，15x15
    in_ctx = np.zeros((N, N), bool)
    in_ctx[ctx_r0:ctx_r1, ctx_c0:ctx_c1] = True
    kept = in_ctx & (idx < 0)

    fig = plt.figure(figsize=(6.8, 2.95))
    axes = [fig.add_axes([0.030, 0.20, 0.255, 0.630]),
            fig.add_axes([0.372, 0.20, 0.255, 0.630]),
            fig.add_axes([0.714, 0.20, 0.255, 0.630])]
    for a in axes:
        a.set_xlim(0, N); a.set_ylim(0, N); a.axis("off")

    # (a) 原图：一个杯子的剪影
    a0 = axes[0]
    for i in range(N):
        for j in range(N):
            ci, cj = i + 0.5, j + 0.5
            d = np.hypot((ci - 8.0) / 4.6, (cj - 8.0) / 4.6)
            fc = "#C9B49C" if d < 1.0 else ("#E4D6C4" if d < 1.5 else "#F2EEE8")
            a0.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="white", lw=0.6))
    a0.set_xlim(0, N); a0.set_ylim(0, N)

    # (b) 目标块
    a1 = axes[1]
    for i in range(N):
        for j in range(N):
            k = idx[i, j]
            fc = "white" if k < 0 else BLOCKS[k][4]
            a1.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="white", lw=0.6,
                                   alpha=1.0 if k >= 0 else 0.85))
    for k, (r0, r1, c0, c1, col) in enumerate(BLOCKS):
        a1.add_patch(Rectangle((c0 + 0.04, N - r1 + 0.04), c1 - c0 - 0.08, r1 - r0 - 0.08,
                               facecolor="none", edgecolor=col, lw=1.3, zorder=5))
        ncell = (r1 - r0) * (c1 - c0)
        a1.text((c0 + c1) / 2, N - (r0 + r1) / 2, f"{ncell / N / N * 100:.0f}%",
                ha="center", va="center", fontsize=7.4, color=fs.INK,
                weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88))

    # (c) 上下文
    a2 = axes[2]
    for i in range(N):
        for j in range(N):
            if kept[i, j]:
                fc = "#BFDBFE"
            elif in_ctx[i, j]:
                fc = "#F7E2D0"
            else:
                fc = "#F4F4F5"
            a2.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="#D4D4D8" if in_ctx[i, j] else "white",
                                   lw=0.6, zorder=3))
    a2.add_patch(Rectangle((ctx_c0 + 0.06, N - ctx_r1 + 0.06), ctx_c1 - ctx_c0 - 0.12,
                           ctx_r1 - ctx_r0 - 0.12, facecolor="none",
                           edgecolor=fs.INK, lw=1.1, ls=(0, (4, 3)), zorder=6))
    for k, (r0, r1, c0, c1, col) in enumerate(BLOCKS):
        a2.add_patch(Rectangle((c0 + 0.08, N - r1 + 0.08), c1 - c0 - 0.16, r1 - r0 - 0.16,
                               facecolor="none", edgecolor=col, lw=0.9,
                               ls=(0, (3, 2)), alpha=0.75, zorder=7))
    a2.text(2.6, 8.5, "可见", ha="center", va="center", fontsize=7.0, color="#1E3A8A",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75), zorder=8)
    a2.text(11.2, 13.2, "挖掉", ha="center", va="center", fontsize=7.0, color="#9A3412",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75), zorder=8)
    a2.text(N / 2, N + 0.6, "上下文大块 $15\\times15$", ha="center", va="bottom",
            fontsize=7.6, color=fs.INK)

    caps = ["（a）原图，切成 16 $\\times$ 16 个 patch",
            "（b）四个目标块，各占 16% 到 19%",
            f"（c）挖去目标区，留下 {kept.sum() / N / N * 100:.0f}% 可见"]
    for x, cap in zip([0.157, 0.500, 0.842], caps):
        fig.text(x, 0.085, cap, ha="center", va="center", fontsize=8.2, color=fs.INK)

    print("target ratios:", [f"{(r1-r0)*(c1-c0)/N/N*100:.1f}%" for r0, r1, c0, c1, _ in BLOCKS],
          "context kept:", f"{kept.sum() / N / N * 100:.1f}%")
    return fs.save(fig, "ch08", "mask-blocks")


# ---------------------------------------------------------------- 图 8-3
def _blob(n, noise=0.0, seed=0, soft=0.0):
    """8x8 的示意图案，模拟一个物体的轮廓。"""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 0.90)
    for i in range(n):
        for j in range(n):
            d = ((i - 3.1) ** 2) / 3.4 + ((j - 3.6) ** 2) / 2.6
            if d < 1:
                img[i, j] = 0.42
            elif d < 1.7:
                img[i, j] = 0.72
    if noise:
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1)
    if soft:
        from scipy.ndimage import uniform_filter  # noqa: F401  (可能不可用)
    return img


def fig_pixel_vs_latent():
    fig = plt.figure(figsize=(7.0, 3.7))
    ax = fig.add_axes([0.0, 0.0, 0.695, 1.0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    n = 6
    gw = 0.155          # 网格在 x 方向上的宽度（y 方向高 0.21，按坐标轴长宽比换算）
    gh = 0.21

    # ---------- (a) 像素空间 ----------
    ax.text(0.020, 0.935, "（a）像素空间的目标", fontsize=8.8, color=fs.GOLD,
            weight="bold", ha="left", va="center")
    y_a = 0.64
    grid_cells(ax, 0.020, y_a, gw, gh, n, "#F0EBE3")
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]:
        cw = gw / n; ch = gh / n
        ax.add_patch(Rectangle((0.020 + j * cw, y_a + (n - 1 - i) * ch), cw, ch,
                               facecolor="#DFD2BF", edgecolor="white", lw=0.6, zorder=4))
    fs.label(ax, 0.020 + gw / 2, y_a - 0.035, "上下文块", fs=7.8, color=fs.MUTED)

    fs.box(ax, 0.215, y_a + 0.015, 0.100, gh - 0.03, "编码器", fc="white", ec=fs.INK,
           fs=8.2, lw=1.0)
    fs.box(ax, 0.345, y_a + 0.015, 0.100, gh - 0.03, "解码器", fc="white", ec=fs.GOLD,
           fs=8.2, lw=1.2)

    # 解码器输出（带噪声的像素）+ 原图像素
    out = _blob(n, noise=0.16, seed=3)
    tar = _blob(n, noise=0.0)
    cw = gw / n; ch = gh / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((0.485 + j * cw, y_a + (n - 1 - i) * ch), cw, ch,
                                   facecolor=str(out[i, j]), edgecolor="white", lw=0.6, zorder=4))
            ax.add_patch(Rectangle((0.705 + j * cw, y_a + (n - 1 - i) * ch), cw, ch,
                                   facecolor=str(tar[i, j]), edgecolor="white", lw=0.6, zorder=4))
    fs.label(ax, 0.485 + gw / 2, y_a - 0.035, "还原出的像素", fs=7.8, color=fs.GOLD)
    fs.label(ax, 0.705 + gw / 2, y_a - 0.035, "原图像素", fs=7.8, color=fs.MUTED)

    fs.arrow(ax, (0.180, y_a + gh / 2), (0.212, y_a + gh / 2), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.318, y_a + gh / 2), (0.342, y_a + gh / 2), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.448, y_a + gh / 2), (0.482, y_a + gh / 2), color=fs.MUTED, lw=1.0)
    ax.annotate("", xy=(0.700, y_a + gh / 2), xytext=(0.645, y_a + gh / 2),
                arrowprops=dict(arrowstyle="<->", color=fs.GOLD, lw=1.0))
    fs.label(ax, 0.672, y_a + gh + 0.045, "逐像素比", fs=7.4, color=fs.GOLD)


    # ---------- (b) 表征空间 ----------
    ax.text(0.020, 0.455, "（b）表征空间的目标", fontsize=8.8, color=fs.ACCENT,
            weight="bold", ha="left", va="center")
    y_b = 0.13
    grid_cells(ax, 0.020, y_b, gw, gh, n, "#F0EBE3")
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]:
        ax.add_patch(Rectangle((0.020 + j * (gw / n), y_b + (n - 1 - i) * (gh / n)),
                               gw / n, gh / n, facecolor="#DFD2BF", edgecolor="white",
                               lw=0.6, zorder=4))
    fs.label(ax, 0.020 + gw / 2, y_b - 0.035, "上下文块", fs=7.8, color=fs.MUTED)

    fs.box(ax, 0.215, y_b + 0.015, 0.100, gh - 0.03, "编码器", fc="white", ec=fs.INK,
           fs=8.2, lw=1.0)
    fs.box(ax, 0.345, y_b + 0.015, 0.100, gh - 0.03, "预测器", fc="white", ec=fs.ACCENT,
           fs=8.2, lw=1.2)
    fs.arrow(ax, (0.180, y_b + gh / 2), (0.212, y_b + gh / 2), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.318, y_b + gh / 2), (0.342, y_b + gh / 2), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.448, y_b + gh / 2), (0.482, y_b + gh / 2), color=fs.MUTED, lw=1.0)

    # 两条向量：预测 vs 目标
    rng = np.random.default_rng(7)
    va = np.clip(rng.normal(0.55, 0.18, 8), 0.1, 0.95)
    vb = np.clip(va + rng.normal(0, 0.09, 8), 0.1, 0.95)
    bw = 0.030
    for k in range(8):
        ax.add_patch(Rectangle((0.500, y_b + 0.004 + k * (gh - 0.008) / 8), bw,
                               (gh - 0.008) / 8, facecolor=str(va[k]),
                               edgecolor="white", lw=0.6, zorder=4))
        ax.add_patch(Rectangle((0.660, y_b + 0.004 + k * (gh - 0.008) / 8), bw,
                               (gh - 0.008) / 8, facecolor=str(vb[k]),
                               edgecolor="white", lw=0.6, zorder=4))
    fs.label(ax, 0.500 + bw / 2, y_b - 0.035, "预测 $\\hat{s}_y$", fs=7.8, color=fs.ACCENT)
    fs.label(ax, 0.660 + bw / 2, y_b - 0.035, "目标 $s_y$", fs=7.8, color=fs.MUTED)
    ax.annotate("", xy=(0.655, y_b + gh / 2), xytext=(0.535, y_b + gh / 2),
                arrowprops=dict(arrowstyle="<->", color=fs.ACCENT, lw=1.0))
    fs.label(ax, 0.595, y_b + gh + 0.045, "在向量上比", fs=7.4, color=fs.ACCENT)

    # ---------- (c) 柱状对比 ----------
    axb = fig.add_axes([0.760, 0.235, 0.225, 0.545])
    vals = [40.7, 66.9]
    axb.bar([0, 1], vals, color=[fs.FAINT, fs.ACCENT], width=0.55, zorder=3)
    for x, v in zip([0, 1], vals):
        axb.text(x, v + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=9.0,
                 color=fs.INK)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["像素目标", "表征目标"], fontsize=8.2)
    axb.set_ylim(0, 82)
    axb.set_yticks([0, 20, 40, 60, 80])
    axb.set_ylabel("1% 标签线性探测 top-1", fontsize=8.0, labelpad=3)
    axb.tick_params(labelsize=7.4)
    fs.clean(axb, spines=("left", "bottom"))
    axb.set_title("（c）两种目标下的探测精度", fontsize=8.2, color=fs.INK, pad=8)
    fig.text(0.872, 0.075, "ViT-L/16，同一套掩码", ha="center", va="center",
             fontsize=7.4, color=fs.MUTED)
    return fs.save(fig, "ch08", "pixel-vs-latent")


if __name__ == "__main__":
    fig_three_networks()
    fig_mask_blocks()
    fig_pixel_vs_latent()
