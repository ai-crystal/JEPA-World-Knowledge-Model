# -*- coding: utf-8 -*-
"""第 10 章插图：世界模型闭环、Dreamer 的隐空间想象。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import figstyle as fs


def fig_loop():
    """图 10-1 世界模型闭环。"""
    fig = plt.figure(figsize=(6.9, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 第一行：环境 → 编码器 → 状态 → 动力学
    y1, h = 0.66, 0.15
    fs.box(ax, 0.045, y1, 0.150, h, "环境\n真实世界", fc="#FBF7F0", ec=fs.INK, fs=9.2)
    fs.box(ax, 0.245, y1, 0.125, h, "编码器\n$E$", fc="#FFFFFF", ec=fs.BLUE, fs=9.2, lw=1.1)
    fs.box(ax, 0.415, y1, 0.125, h, "状态\n$s_t$", fc="#F2F6FF", ec=fs.BLUE, fs=9.4, lw=1.2)
    fs.box(ax, 0.585, y1, 0.165, h, "动力学\n$f_\\theta(s_t,a_t)$", fc="#FFFFFF",
           ec=fs.ACCENT, fs=9.2, lw=1.2)

    fs.arrow(ax, (0.197, 0.735), (0.242, 0.735), color=fs.INK, lw=1.1)
    fs.arrow(ax, (0.372, 0.735), (0.412, 0.735), color=fs.INK, lw=1.1)
    fs.arrow(ax, (0.542, 0.735), (0.582, 0.735), color=fs.INK, lw=1.1)
    fs.label(ax, 0.219, 0.625, "观测 $x_t$", fs=8.4, color=fs.MUTED)
    fs.label(ax, 0.392, 0.625, "压成向量", fs=8.4, color=fs.MUTED)
    fs.label(ax, 0.562, 0.625, "预测下一状态", fs=8.4, color=fs.MUTED)

    # 预测结果回灌：在状态上连续推进
    fs.arrow(ax, (0.735, 0.812), (0.492, 0.816), color=fs.ACCENT, lw=1.1,
             ls=(0, (4, 3)), rad=0.45, ms=9, z=6)
    fs.label(ax, 0.605, 0.962, "把预测值喂回，在内部连续推进", fs=8.2, color=fs.ACCENT)

    # 第二行：策略 → 动作
    y2 = 0.38
    fs.box(ax, 0.415, y2, 0.125, h, "策略\n$\\pi$", fc="#FFFFFF", ec=fs.GREEN, fs=9.4, lw=1.1)
    fs.box(ax, 0.585, y2, 0.165, h, "动作\n$a_t$", fc="#F0FBF6", ec=fs.GREEN, fs=9.4, lw=1.2)
    fs.arrow(ax, (0.4775, 0.658), (0.4775, 0.533), color=fs.INK, lw=1.1)
    fs.arrow(ax, (0.542, 0.455), (0.582, 0.455), color=fs.INK, lw=1.1)
    fs.label(ax, 0.466, 0.596, "读状态", fs=8.4, color=fs.MUTED, ha="right")
    fs.arrow(ax, (0.6675, 0.533), (0.6675, 0.658), color=fs.ACCENT, lw=1.0,
             ls=(0, (3, 3)), ms=8)
    fs.label(ax, 0.686, 0.596, "条件输入", fs=8.2, color=fs.ACCENT, ha="left")

    # 动作执行后回到环境
    ax.plot([0.6675, 0.6675], [0.378, 0.215], color=fs.FAINT, lw=1.0, zorder=2)
    ax.plot([0.6675, 0.120], [0.215, 0.215], color=fs.FAINT, lw=1.0, zorder=2)
    fs.arrow(ax, (0.120, 0.215), (0.120, 0.658), color=fs.MUTED, lw=1.1, ms=10)
    fs.label(ax, 0.400, 0.150, "执行动作，环境返回下一帧观测 $x_{t+1}$", fs=8.4, color=fs.MUTED)
    return fs.save(fig, "ch10", "world-model-loop")


def fig_imagine():
    """图 10-2 真实轨迹与想象轨迹。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.35))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.855, wspace=0.16)

    for ax in axes:
        ax.set_xlim(0.2, 8.6)
        ax.set_ylim(0.4, 8.9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("潜空间的第一维（示意）", fontsize=8.4, color=fs.MUTED)
        ax.set_ylabel("第二维", fontsize=8.4, color=fs.MUTED)
        fs.clean(ax, spines=("left", "bottom"))

    # (a) 真实轨迹：编码器对回放片段逐帧编码
    a = axes[0]
    real = np.array([[0.9, 2.0], [2.1, 2.7], [3.5, 3.2], [4.9, 4.1], [6.0, 5.0]])
    a.plot(real[:, 0], real[:, 1], color=fs.BLUE, lw=1.8, zorder=3, solid_capstyle="round")
    for i in range(len(real) - 1):
        p0, p1 = real[i], real[i + 1]
        d = p1 - p0
        fs.arrow(a, tuple(p0 + d * 0.62), tuple(p0 + d * 0.80), color=fs.BLUE, lw=1.2, ms=9)
    a.scatter(real[:, 0], real[:, 1], s=46, color=fs.BLUE, zorder=4,
              edgecolor="white", linewidth=1.2)
    for i, (x, y) in enumerate(real):
        a.text(x, y - 0.62, f"$s_{i + 1}$", fontsize=8.8, color=fs.BLUE,
               ha="center", va="top", zorder=5)
    a.set_title("（a）回放缓冲里的真实轨迹", fontsize=9.6, color=fs.INK, pad=8)
    a.text(0.30, 7.9, "观测逐帧过编码器", fontsize=8.4, color=fs.MUTED, ha="left")

    # (b) 想象轨迹：从同一状态出发，策略采样动作，动力学推进
    b = axes[1]
    start = np.array([1.6, 4.4])
    paths = [
        (fs.ACCENT, [[2.7, 5.2], [3.9, 5.7], [5.1, 6.3], [6.3, 6.7], [7.4, 7.4]]),
        (fs.PURPLE, [[2.8, 4.0], [4.0, 3.4], [5.2, 3.0], [6.4, 2.5], [7.5, 2.4]]),
        (fs.GREEN,  [[2.6, 4.5], [3.8, 4.6], [5.0, 4.1], [6.2, 4.7], [7.3, 4.9]]),
    ]
    for col, pts in paths:
        p = np.vstack([start, np.array(pts)])
        b.plot(p[:, 0], p[:, 1], color=col, lw=1.4, ls=(0, (4, 2.5)), zorder=3)
        b.scatter(p[1:, 0], p[1:, 1], s=20, facecolor="white", edgecolor=col,
                  linewidth=1.1, zorder=4)
        b.scatter([p[-1, 0]], [p[-1, 1]], s=32, color=col, zorder=4,
                  edgecolor="white", linewidth=1.1)
    b.scatter([start[0]], [start[1]], s=64, color=fs.INK, zorder=5,
              edgecolor="white", linewidth=1.3)
    b.text(start[0] - 0.12, start[1] + 0.52, "$s_t$", fontsize=9.4, color=fs.INK, ha="left")
    b.set_title("（b）从同一状态出发的想象轨迹", fontsize=9.6, color=fs.INK, pad=8)
    b.text(0.30, 7.9, "同一状态、不同动作序列", fontsize=8.4, color=fs.MUTED, ha="left")
    b.annotate("", xy=(7.5, 1.55), xytext=(2.6, 1.55),
               arrowprops=dict(arrowstyle="<->", color=fs.FAINT, lw=0.9))
    b.text(5.05, 1.12, "在动力学里推进 $T=16$ 步", fontsize=8.4, color=fs.MUTED, ha="center")
    return fs.save(fig, "ch10", "dreamer-imagination")


if __name__ == "__main__":
    fig_loop()
    fig_imagine()
