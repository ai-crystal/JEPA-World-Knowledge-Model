# -*- coding: utf-8 -*-
"""第 13 章插图：三种解释的因果图、校准前后的相似度对比（定性示意）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs


def fig_three_explanations():
    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 左侧三条可能的原因
    causes = [
        ("容量", "假设空间够大\n才能覆盖同一个最优解", fs.BLUE, 0.690),
        ("任务", "约束越多\n满足全部约束的解越少", fs.GREEN, 0.450),
        ("数据", "同一个世界\n同一套统计结构", fs.PURPLE, 0.210),
    ]
    for name, sub, col, y in causes:
        fs.box(ax, 0.035, y, 0.215, 0.160, "", fc="#FFFFFF", ec=col, lw=1.2)
        fs.label(ax, 0.062, y + 0.118, name, fs=10.5, weight="bold", color=col, ha="left")
        fs.label(ax, 0.062, y + 0.092, sub, fs=7.4, color=fs.MUTED, ha="left", va="top")

    # 中间：被解释的对象
    fs.box(ax, 0.365, 0.40, 0.195, 0.20, "", fc="#F7F4EE", ec=fs.INK, lw=1.2)
    fs.label(ax, 0.4625, 0.525, "共享的", fs=9.6, color=fs.INK)
    fs.label(ax, 0.4625, 0.465, "世界结构", fs=9.6, color=fs.INK)
    fs.label(ax, 0.4625, 0.412, "同一个解", fs=7.6, color=fs.MUTED)

    # 右侧：实际上被测量的东西
    fs.box(ax, 0.665, 0.40, 0.255, 0.20, "", fc="#FFFFFF", ec=fs.ACCENT, lw=1.3)
    fs.label(ax, 0.7925, 0.525, "测得的", fs=9.6, color=fs.ACCENT)
    fs.label(ax, 0.7925, 0.465, "相似度分数", fs=9.6, color=fs.ACCENT)
    fs.label(ax, 0.7925, 0.412, "CKA / mKNN", fs=7.6, color=fs.MUTED)

    # 三条原因汇入中间
    fs.arrow(ax, (0.253, 0.770), (0.362, 0.575), color=fs.BLUE, lw=1.1, rad=-0.10)
    fs.arrow(ax, (0.253, 0.530), (0.362, 0.500), color=fs.GREEN, lw=1.1)
    fs.arrow(ax, (0.253, 0.290), (0.362, 0.425), color=fs.PURPLE, lw=1.1, rad=0.10)

    # 中间 → 测量
    fs.arrow(ax, (0.563, 0.500), (0.662, 0.500), color=fs.INK, lw=1.2)

    # 下方的测量环节，虚线通向分数
    fs.box(ax, 0.365, 0.060, 0.245, 0.115, "", fc="#FFFFFF", ec=fs.FAINT, lw=1.0, rounded=0.015)
    fs.label(ax, 0.4875, 0.135, "测量环节", fs=9.0, color=fs.MUTED)
    fs.label(ax, 0.4875, 0.092, "宽度、深度、层对取最大", fs=7.6, color=fs.MUTED)
    fs.arrow(ax, (0.612, 0.150), (0.742, 0.397), color=fs.FAINT, lw=1.1,
             ls=(0, (4, 3)), rad=-0.06)
    fs.label(ax, 0.575, 0.235, "伪影通道", fs=7.8, color=fs.MUTED, ha="right")

    # 左边的图注性说明
    fs.label(ax, 0.035, 0.955, "可能的成因", fs=8.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.035, 0.915, "三者并不互斥", fs=8.4, color=fs.MUTED, ha="left")
    return fs.save(fig, "ch13", "three-explanations")


def fig_calibrated():
    fig = plt.figure(figsize=(6.6, 2.8))
    ax = fig.add_axes([0.20, 0.11, 0.76, 0.76])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.34, 1.52)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    rows = [
        ("全局谱指标\nCKA", fs.ACCENT, 1.0, 0.82, 0.11, "大幅下降"),
        ("局部近邻\nmKNN", fs.BLUE, 0.0, 0.82, 0.65, "基本保留"),
    ]

    # 零基线
    ax.plot([0.035, 0.035], [-0.24, 1.40], color=fs.FAINT, lw=0.9, ls=(0, (3, 3)), zorder=1)
    fs.label(ax, 0.050, 1.40, "零基线", fs=7.8, color=fs.MUTED, ha="left")

    for name, col, y, before, after, note in rows:
        # 行底色
        ax.add_patch(Rectangle((0.0, y - 0.30), 1.0, 0.60,
                               facecolor="#F8FAFC" if y > 0.5 else "#FFFFFF",
                               edgecolor="none", zorder=0))
        # 连接线
        ax.plot([after, before], [y, y], color=col, lw=6, alpha=0.22,
                solid_capstyle="round", zorder=2)
        # 两个端点
        ax.scatter([before], [y], s=70, facecolor="white", edgecolor=col,
                   linewidth=1.5, zorder=4)
        ax.scatter([after], [y], s=70, facecolor=col, edgecolor=col,
                   linewidth=1.5, zorder=4)
        # 方向箭头
        fs.arrow(ax, (before - 0.015, y + 0.115), (after + 0.015, y + 0.115),
                 color=col, lw=1.0, ms=7)
        fs.label(ax, (before + after) / 2, y + 0.315, note, fs=8.2, color=col)
        # 端点标注
        fs.label(ax, before, y - 0.135, "校准前", fs=7.8, color=fs.MUTED)
        fs.label(ax, after, y - 0.135, "校准后", fs=7.8, color=fs.MUTED)
        # 行名
        fs.label(ax, -0.035, y, name, fs=9.2, color=fs.INK, ha="right")

    return fs.save(fig, "ch13", "calibrated")


if __name__ == "__main__":
    fig_three_explanations()
    fig_calibrated()
