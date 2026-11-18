# -*- coding: utf-8 -*-
"""第 1 章插图：三条选择性压力与一次测量、校准前后的相关系数、结构-几何 2x2 框架。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs


def fig_three_pressures():
    fig = plt.figure(figsize=(7.0, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    causes = [
        ("任务通用性", "解的任务越多\n可行解越少", fs.GREEN, 0.700),
        ("容量", "假设空间越大\n越容易盖住最优解", fs.BLUE, 0.460),
        ("简单性偏置", "模型越大\n越偏向简单的拟合", fs.PURPLE, 0.220),
    ]
    for name, sub, col, y in causes:
        fs.box(ax, 0.030, y, 0.225, 0.155, "", fc="#FFFFFF", ec=col, lw=1.2)
        fs.label(ax, 0.056, y + 0.112, name, fs=10.2, weight="bold", color=col, ha="left")
        fs.label(ax, 0.056, y + 0.086, sub, fs=7.4, color=fs.MUTED, ha="left", va="top")

    # 中间：被解释的对象
    fs.box(ax, 0.368, 0.415, 0.190, 0.185, "", fc="#F7F4EE", ec=fs.INK, lw=1.2)
    fs.label(ax, 0.463, 0.545, "共享的", fs=9.4, color=fs.INK)
    fs.label(ax, 0.463, 0.492, "世界结构", fs=9.4, color=fs.INK)
    fs.label(ax, 0.463, 0.438, "同一个解", fs=7.4, color=fs.MUTED)

    # 右侧：实际上被测量的东西
    fs.box(ax, 0.668, 0.415, 0.255, 0.185, "", fc="#FFFFFF", ec=fs.ACCENT, lw=1.3)
    fs.label(ax, 0.7955, 0.545, "测得的", fs=9.4, color=fs.ACCENT)
    fs.label(ax, 0.7955, 0.492, "相似度分数", fs=9.4, color=fs.ACCENT)
    fs.label(ax, 0.7955, 0.438, "CKA / mKNN", fs=7.4, color=fs.MUTED)

    fs.arrow(ax, (0.258, 0.778), (0.365, 0.590), color=fs.GREEN, lw=1.1, rad=-0.10)
    fs.arrow(ax, (0.258, 0.538), (0.365, 0.512), color=fs.BLUE, lw=1.1)
    fs.arrow(ax, (0.258, 0.298), (0.365, 0.435), color=fs.PURPLE, lw=1.1, rad=0.10)
    fs.arrow(ax, (0.560, 0.508), (0.665, 0.508), color=fs.INK, lw=1.2)

    # 下方的测量环节，虚线通向分数
    fs.box(ax, 0.368, 0.105, 0.245, 0.110, "", fc="#FFFFFF", ec=fs.FAINT, lw=1.0, rounded=0.015)
    fs.label(ax, 0.4905, 0.177, "测量环节", fs=8.8, color=fs.MUTED)
    fs.label(ax, 0.4905, 0.133, "宽度、深度、层对取最大", fs=7.4, color=fs.MUTED)
    fs.arrow(ax, (0.615, 0.192), (0.745, 0.412), color=fs.FAINT, lw=1.1,
             ls=(0, (4, 3)), rad=-0.06)
    fs.label(ax, 0.592, 0.268, "伪影通道", fs=7.6, color=fs.MUTED, ha="right")

    fs.label(ax, 0.030, 0.960, "可能的成因", fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax, 0.030, 0.922, "三者并不互斥", fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax, 0.030, 0.045, "三者的共同前提：所有模型都在同一个世界上取样",
             fs=7.6, color=fs.MUTED, ha="left")
    return fs.save(fig, "theory", "ch01", "three-pressures")


def fig_calibrated():
    """横轴是分数与模型能力排名的相关系数，左空心点是校准前，右实心点是校准后。"""
    fig = plt.figure(figsize=(6.6, 3.0))
    ax = fig.add_axes([0.235, 0.20, 0.715, 0.66])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.62, 3.72)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(fs.FAINT)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("与语言模型能力排名的相关系数", fontsize=8.6, color=fs.MUTED, labelpad=5)
    ax.tick_params(axis="x", colors=fs.MUTED, labelsize=8.0, length=3)

    rows = [
        ("线性 CKA", fs.ACCENT, 3, 0.86, 0.45, "0.86", "0.45"),
        ("普罗克鲁斯提斯距离", fs.GOLD, 2, 0.89, 0.39, "0.89", "0.39"),
        ("互近邻 mKNN", fs.BLUE, 1, None, 0.85, None, "0.85"),
        ("CKNNA", fs.TEAL, 0, None, 0.87, None, "0.87"),
    ]

    for name, col, y, before, after, lb, la in rows:
        if y % 2 == 1:
            ax.add_patch(Rectangle((0.0, y - 0.42), 1.0, 0.84,
                                   facecolor="#F8FAFC", edgecolor="none", zorder=0))
        if before is None:
            # 校准前后几乎重合，只标一个点
            ax.plot([after, after], [y - 0.22, y + 0.22], color=col, lw=5, alpha=0.20,
                    solid_capstyle="round", zorder=2)
            ax.scatter([after], [y], s=62, facecolor=col, edgecolor=col, linewidth=1.5, zorder=4)
            fs.label(ax, after, y + 0.30, la, fs=8.4, color=col)
            fs.label(ax, after, y - 0.34, "前后几乎重合", fs=7.4, color=fs.MUTED)
        else:
            ax.plot([after, before], [y, y], color=col, lw=6, alpha=0.20,
                    solid_capstyle="round", zorder=2)
            ax.scatter([before], [y], s=62, facecolor="white", edgecolor=col,
                       linewidth=1.5, zorder=4)
            ax.scatter([after], [y], s=62, facecolor=col, edgecolor=col,
                       linewidth=1.5, zorder=4)
            fs.arrow(ax, (before - 0.02, y + 0.20), (after + 0.02, y + 0.20),
                     color=col, lw=0.9, ms=7)
            fs.label(ax, before, y + 0.42, lb, fs=8.4, color=col)
            fs.label(ax, after, y + 0.42, la, fs=8.4, color=col)
            fs.label(ax, before, y - 0.34, "校准前", fs=7.4, color=fs.MUTED)
            fs.label(ax, after, y - 0.34, "校准后", fs=7.4, color=fs.MUTED)
        fs.label(ax, -0.035, y, name, fs=8.6, color=fs.INK, ha="right")

    return fs.save(fig, "theory", "ch01", "calibrated")


def fig_structure_geometry():
    fig = plt.figure(figsize=(6.7, 4.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    W = 0.378
    LX, GX = 0.150, 0.556
    fs.label(ax, LX + W / 2, 0.955, "局部", fs=10.4, color=fs.INK, weight="bold")
    fs.label(ax, GX + W / 2, 0.955, "全局", fs=10.4, color=fs.INK, weight="bold")

    fs.label(ax, 0.083, 0.730, "关系结构", fs=10.2, color=fs.GREEN, weight="bold", rotation=90)
    fs.label(ax, 0.083, 0.375, "度量几何", fs=10.2, color=fs.ACCENT, weight="bold", rotation=90)

    cells = [
        (LX, 0.655, fs.GREEN, "#F1F7F3",
         "互近邻 mKNN", "数两边近邻集合的重合比例", "204/204 对显著"),
        (GX, 0.655, fs.GREEN, "#F1F7F3",
         "骨架重合 $H_0$", "最小生成树共有边占 $n-1$ 的比例", "204/204 对显著"),
        (LX, 0.300, fs.ACCENT, "#FFF7F2",
         "距离感知 mKNN", "在此基础上追加距离接近的要求", "τ = 1 保留 85.6%"),
        (GX, 0.300, fs.ACCENT, "#FFF7F2",
         "距离感知骨架", "距离按各自第 90 百分位归一化", "τ = 1 保留 83.0%"),
    ]
    for x, y, col, fc, title, sub, num in cells:
        fs.box(ax, x, y, W, 0.150, "", fc=fc, ec=col, lw=1.2)
        fs.label(ax, x + 0.016, y + 0.112, title, fs=9.6, color=col, ha="left", weight="bold")
        fs.label(ax, x + 0.016, y + 0.070, sub, fs=7.4, color=fs.MUTED, ha="left")
        fs.label(ax, x + 0.016, y + 0.030, num, fs=8.0, color=fs.INK, ha="left")

    fs.label(ax, LX, 0.610, "校准后仍然显著，且随能力上升", fs=7.6, color=fs.MUTED, ha="left")
    fs.label(ax, LX, 0.255, "越要求距离一致，对齐衰减越快", fs=7.6, color=fs.MUTED, ha="left")

    fs.arrow(ax, (LX + W / 2, 0.232), (LX + W / 2, 0.178), color=fs.ACCENT, lw=1.0, ms=7)
    fs.label(ax, LX + W / 2, 0.155, "τ = $10^{-2}$ 时只剩 5.8%", fs=7.8, color=fs.ACCENT)
    fs.arrow(ax, (GX + W / 2, 0.232), (GX + W / 2, 0.178), color=fs.ACCENT, lw=1.0, ms=7)
    fs.label(ax, GX + W / 2, 0.155, "τ = $10^{-2}$ 时只剩 5.5%", fs=7.8, color=fs.ACCENT)

    fs.label(ax, LX, 0.078, "最严一档（τ = $10^{-4}$）仍显著的比例：局部 89.7%，全局 40.2%",
             fs=7.9, color=fs.INK, ha="left")
    return fs.save(fig, "theory", "ch01", "structure-geometry")


if __name__ == "__main__":
    fig_three_pressures()
    fig_calibrated()
    fig_structure_geometry()
