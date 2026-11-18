# -*- coding: utf-8 -*-
"""第 3 章插图：三条自监督路线的结构对比、三种目标各自的拉近关系。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs

# 三联图共用同一套列位置，保证三张子图的读法一致
COL_A, COL_B, COL_P, COL_T = 0.01, 0.24, 0.46, 0.69


def _target_space(ax, name, y0, h):
    """目标空间底纹：三条路线都把“损失落在哪个空间”画在同一个位置。"""
    ax.add_patch(Rectangle((0.67, y0), 0.32, h, facecolor=fs.LIGHT,
                           edgecolor="none", zorder=0))
    fs.label(ax, 0.83, y0 + h + 0.035, "目标空间　" + name, fs=7.0, color=fs.MUTED)


def _a(ax, p0, p1, color=fs.INK, ms=5.5, lw=0.95):
    fs.arrow(ax, p0, p1, color=color, lw=lw, ms=ms)


def fig_three_routes():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.85))
    fig.subplots_adjust(left=0.008, right=0.992, top=0.99, bottom=0.01, wspace=0.05)
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0.06, 1.0); ax.axis("off")

    # ---------------- (a) 生成式 ----------------
    ax = axes[0]
    fs.label(ax, 0.0, 0.955, "(a) 生成式", fs=9.0, weight="bold", ha="left", color=fs.ACCENT)
    _target_space(ax, "像素", 0.335, 0.490)
    fs.box(ax, 0.01, 0.58, 0.16, 0.16, "图像", fc="white", ec=fs.INK, fs=7.6)
    fs.box(ax, 0.24, 0.68, 0.16, 0.14, "编码器", fc="white", ec=fs.BLUE, fs=7.4)
    fs.box(ax, 0.24, 0.42, 0.16, 0.14, "解码器", fc="white", ec=fs.BLUE, fs=7.4)
    fs.box(ax, 0.69, 0.42, 0.22, 0.14, "重建 $\\hat{x}$", fc="white", ec=fs.ACCENT, fs=7.6)
    _a(ax, (0.174, 0.665), (0.236, 0.735))
    _a(ax, (0.320, 0.678), (0.320, 0.564))
    _a(ax, (0.404, 0.490), (0.684, 0.490))
    # 像素级损失：从输入绕到底部，再回到重建
    ax.plot([0.090, 0.090], [0.578, 0.215], color=fs.ACCENT, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.plot([0.090, 0.800], [0.215, 0.215], color=fs.ACCENT, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.plot([0.800, 0.800], [0.215, 0.382], color=fs.ACCENT, lw=1.0, ls=(0, (4, 3)), zorder=2)
    fs.arrow(ax, (0.800, 0.382), (0.800, 0.418), color=fs.ACCENT, lw=1.0, ms=5.5)
    fs.label(ax, 0.440, 0.145, "逐像素算差", fs=7.2, color=fs.ACCENT)

    # ---------------- (b) 对比式 ----------------
    ax = axes[1]
    fs.label(ax, 0.0, 0.955, "(b) 对比式", fs=9.0, weight="bold", ha="left", color=fs.GREEN)
    _target_space(ax, "嵌入", 0.335, 0.490)
    fs.box(ax, 0.01, 0.64, 0.16, 0.14, "视图\n$x_1$", fc="white", ec=fs.INK, fs=7.2)
    fs.box(ax, 0.01, 0.40, 0.16, 0.14, "视图\n$x_2$", fc="white", ec=fs.INK, fs=7.2)
    fs.box(ax, 0.24, 0.40, 0.16, 0.38, "共享\n编码器", fc="white", ec=fs.BLUE, fs=7.2)
    fs.box(ax, 0.69, 0.64, 0.22, 0.13, "$z_1$", fc="white", ec=fs.GREEN, fs=8.2)
    fs.box(ax, 0.69, 0.41, 0.22, 0.13, "$z_2$", fc="white", ec=fs.GREEN, fs=8.2)
    _a(ax, (0.174, 0.705), (0.236, 0.685))
    _a(ax, (0.174, 0.475), (0.236, 0.495))
    _a(ax, (0.404, 0.685), (0.684, 0.705))
    _a(ax, (0.404, 0.495), (0.684, 0.475))
    fs.arrow(ax, (0.800, 0.636), (0.800, 0.547), color=fs.GREEN, lw=1.2, style="<|-|>", ms=5.5)
    fs.label(ax, 0.866, 0.588, "拉近", fs=7.0, color=fs.GREEN, ha="left")
    # 负样本：同一批里的其他样本，被推远
    for xn in (0.720, 0.800, 0.880):
        ax.add_patch(plt.Circle((xn, 0.235), 0.021, facecolor="white",
                                edgecolor=fs.FAINT, lw=0.9, zorder=3))
        fs.arrow(ax, (0.800, 0.406), (xn, 0.260), color=fs.FAINT, lw=0.9,
                 style="-|>", ls=(0, (3, 2)), ms=4.5)
    fs.label(ax, 0.800, 0.158, "其他样本", fs=6.9, color=fs.MUTED)
    fs.label(ax, 0.620, 0.232, "推远", fs=7.0, color=fs.ACCENT)

    # ---------------- (c) 联合嵌入预测 ----------------
    ax = axes[2]
    fs.label(ax, 0.0, 0.955, "(c) 联合嵌入预测", fs=9.0, weight="bold", ha="left",
             color=fs.PURPLE)
    _target_space(ax, "嵌入", 0.335, 0.490)
    fs.box(ax, 0.01, 0.62, 0.16, 0.14, "上下文", fc="white", ec=fs.INK, fs=7.4)
    fs.box(ax, 0.01, 0.38, 0.16, 0.14, "目标块", fc="white", ec=fs.INK, fs=7.4)
    fs.box(ax, 0.22, 0.62, 0.16, 0.14, "上下文\n编码器", fc="white", ec=fs.BLUE, fs=6.9)
    fs.box(ax, 0.22, 0.38, 0.16, 0.14, "目标\n编码器", fc="white", ec=fs.BLUE, fs=6.9)
    fs.box(ax, 0.44, 0.62, 0.16, 0.14, "预测器", fc="white", ec=fs.PURPLE, fs=6.9)
    fs.box(ax, 0.69, 0.63, 0.22, 0.13, "$\\hat{z}_y$", fc="white", ec=fs.PURPLE, fs=8.4)
    fs.box(ax, 0.69, 0.39, 0.22, 0.13, "$z_y$", fc="white", ec=fs.PURPLE, fs=8.4)
    _a(ax, (0.174, 0.690), (0.216, 0.690))
    _a(ax, (0.174, 0.450), (0.216, 0.450))
    _a(ax, (0.384, 0.690), (0.434, 0.690), color=fs.PURPLE, ms=5.0)
    _a(ax, (0.604, 0.690), (0.684, 0.694), color=fs.PURPLE)
    _a(ax, (0.384, 0.450), (0.684, 0.456))
    fs.arrow(ax, (0.800, 0.626), (0.800, 0.526), color=fs.ACCENT, lw=1.2, style="<|-|>", ms=5.5)
    fs.label(ax, 0.866, 0.570, "L2", fs=7.2, color=fs.ACCENT, ha="left")
    fs.label(ax, 0.320, 0.310, "权重由 EMA 更新", fs=6.8, color=fs.MUTED)

    return fs.save(fig, "basics", "ch03", "three-routes")


def _sample_square(ax, cx, cy, w=0.087, h=0.072, ec=fs.INK, fc="white", z=3):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fc,
                           edgecolor=ec, lw=1.0, zorder=z))


def fig_pull_sets():
    """三栏统一版式：四个样本各占一列，(a) 单格自环，(b) 双格上下配对并互相推开，
    (c) 上下两格单向拉近且不推。"""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75))
    fig.subplots_adjust(left=0.008, right=0.992, top=0.99, bottom=0.01, wspace=0.06)
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0.02, 1.0); ax.axis("off")

    xs = [0.155, 0.39, 0.625, 0.86]
    y_top, y_bot = 0.72, 0.36
    notes = ["每个样本还原自己", "同一物体两视图拉近，不同样本推开", "两块拉近，不推开其他"]
    titles = ["(a) 生成式", "(b) 对比式", "(c) 联合嵌入预测"]
    tcol = [fs.ACCENT, fs.GREEN, fs.PURPLE]

    # ---------------- (a) ----------------
    ax = axes[0]
    fs.label(ax, 0.0, 0.955, titles[0], fs=8.8, weight="bold", ha="left", color=tcol[0])
    for i, x in enumerate(xs):
        _sample_square(ax, x, 0.54, w=0.105, h=0.082)
        fs.arrow(ax, (x + 0.056, 0.594), (x - 0.056, 0.594),
                 color=fs.BLUE, lw=1.1, rad=1.05, ms=6.5)
        fs.label(ax, x, 0.40, str(i + 1), fs=7.4, color=fs.MUTED)
    fs.label(ax, 0.5, 0.16, notes[0], fs=7.2, color=fs.MUTED)

    # ---------------- (b) ----------------
    ax = axes[1]
    fs.label(ax, 0.0, 0.955, titles[1], fs=8.8, weight="bold", ha="left", color=tcol[1])
    for i, x in enumerate(xs):
        _sample_square(ax, x, y_top, w=0.088, h=0.078)
        _sample_square(ax, x, y_bot, w=0.088, h=0.078)
        fs.arrow(ax, (x, y_top - 0.05), (x, y_bot + 0.05),
                 color=fs.GREEN, lw=1.2, style="<|-|>", ms=5.5)
        fs.label(ax, x, 0.245, str(i + 1), fs=7.4, color=fs.MUTED)
    for a, b in zip(xs[:-1], xs[1:]):
        for yy in (y_top, y_bot):
            fs.arrow(ax, (a + 0.056, yy), (b - 0.056, yy),
                     color=fs.ACCENT, lw=0.9, style="<|-|>", ls=(0, (2.5, 2)), ms=4.2)
    fs.label(ax, 0.5, 0.16, notes[1], fs=7.0, color=fs.MUTED)

    # ---------------- (c) ----------------
    ax = axes[2]
    fs.label(ax, 0.0, 0.955, titles[2], fs=8.8, weight="bold", ha="left", color=tcol[2])
    for i, x in enumerate(xs):
        _sample_square(ax, x, y_top, w=0.088, h=0.078, ec=fs.INK)
        _sample_square(ax, x, y_bot, w=0.088, h=0.078, ec=fs.PURPLE, fc="#FBF7FF")
        fs.arrow(ax, (x, y_top - 0.05), (x, y_bot + 0.05),
                 color=fs.PURPLE, lw=1.2, style="-|>", ms=5.5)
        fs.label(ax, x, 0.245, str(i + 1), fs=7.4, color=fs.MUTED)
    fs.label(ax, 0.5, 0.16, notes[2], fs=7.0, color=fs.MUTED)

    return fs.save(fig, "basics", "ch03", "pull-sets")


if __name__ == "__main__":
    fig_three_routes()
    fig_pull_sets()
