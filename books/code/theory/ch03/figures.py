# -*- coding: utf-8 -*-
"""第 3 章插图：旋转轨道，以及三条前提被去掉之后最小值集合的形状。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import figstyle as fs


# ---------------------------------------------------------------- 图 3-1
def constellation():
    """一族不规则的点，像一片街区的路口。"""
    pts = np.array([
        [-0.75, 0.05], [-0.30, 0.50], [0.10, 0.30], [0.60, 0.55],
        [0.45, -0.25], [-0.20, -0.60],
    ])
    return pts


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def fig_rotation_orbit():
    """左三格：同一族点在三套坐标下的位置；右一格：一个点的轨道。"""
    fig = plt.figure(figsize=(7.0, 2.62))
    hi = [(0.0, fs.BLUE, r"$Q_1h^\star$", "模态 1"),
          (40.0, fs.ACCENT, r"$Q_2h^\star$", "模态 2"),
          (80.0, fs.GREEN, r"$Q_3h^\star$", "模态 3")]
    P = constellation()
    order = [0, 1, 2, 3, 4, 5, 0]
    boxes = [(0.015, 0.20, 0.225, 0.72), (0.250, 0.20, 0.225, 0.72),
             (0.485, 0.20, 0.225, 0.72)]
    for (x0, y0, w, h), (deg, col, lab, name) in zip(boxes, hi):
        ax = fig.add_axes([x0, y0, w, h])
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        for v in ((1.0, 0.0), (0.0, 1.0)):
            ax.annotate("", xy=v, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=fs.FAINT, lw=0.8,
                                        mutation_scale=7))
        Q = P @ rot(np.deg2rad(deg)).T
        ax.plot(Q[order, 0], Q[order, 1], color=col, lw=0.9, alpha=0.45, zorder=2)
        ax.scatter(Q[:, 0], Q[:, 1], s=26, color=col, zorder=4,
                   edgecolor="white", linewidth=0.7)
        ax.scatter([0], [0], s=6, color=fs.FAINT, zorder=3)
        ax.set_title(lab, fontsize=10.4, color=col, pad=1.5)
        fig.text(x0 + w / 2, 0.055, name, ha="center", va="center",
                 fontsize=8.4, color=fs.MUTED)

    # 右格：轨道
    axo = fig.add_axes([0.723, 0.20, 0.272, 0.72])
    axo.set_xlim(-1.05, 1.05)
    axo.set_ylim(-1.05, 1.05)
    axo.set_aspect("equal")
    axo.axis("off")
    r0 = np.hypot(P[3, 0], P[3, 1])
    th0 = np.arctan2(P[3, 2 - 1], P[3, 0])
    axo.add_patch(Circle((0, 0), r0, facecolor="none", edgecolor=fs.INK,
                         lw=1.0, ls=(0, (3, 3)), zorder=2))
    for deg in range(6, 80, 8):
        a = th0 + np.deg2rad(deg)
        axo.scatter([r0 * np.cos(a)], [r0 * np.sin(a)], s=12, facecolor="white",
                    edgecolor=fs.FAINT, linewidth=0.7, zorder=3)
    axo.plot([0, r0 * np.cos(th0)], [0, r0 * np.sin(th0)], color=fs.FAINT, lw=0.7, zorder=2)
    for deg, col, lab, name in hi:
        a = th0 + np.deg2rad(deg)
        axo.scatter([r0 * np.cos(a)], [r0 * np.sin(a)], s=30, color=col, zorder=5,
                    edgecolor="white", linewidth=0.8)
    tt = np.linspace(th0, th0 + np.deg2rad(80), 70)
    axo.plot(0.36 * np.cos(tt), 0.36 * np.sin(tt), color=fs.ACCENT, lw=0.9, zorder=4)
    axo.add_patch(FancyArrowPatch((0.36 * np.cos(tt[-2]), 0.36 * np.sin(tt[-2])),
                                  (0.36 * np.cos(tt[-1]), 0.36 * np.sin(tt[-1])),
                                  arrowstyle="-|>", mutation_scale=7, color=fs.ACCENT,
                                  lw=0.9, zorder=5))
    fs.label(axo, 0.50 * np.cos(th0 + np.deg2rad(30)), 0.50 * np.sin(th0 + np.deg2rad(30)),
             r"$\theta$", fs=9.0, color=fs.ACCENT)
    axo.scatter([0], [0], s=7, color=fs.INK, zorder=6)
    axo.set_title(r"$O(k)\,h^\star$", fontsize=10.4, color=fs.INK, pad=1.5)
    fig.text(0.859, 0.055, "整条轨道都是最小值点", ha="center", va="center",
             fontsize=8.4, color=fs.MUTED)
    return fs.save(fig, "theory", "ch03", "rotation-orbit")


# ---------------------------------------------------------------- 图 3-2
def panel_frame(ax, xlim=(-1.24, 1.24), ylim=(-1.16, 1.16)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fs.clean(ax, spines=())


def fig_failure_modes():
    """三条前提各自被去掉之后，最小值集合长什么样。"""
    fig = plt.figure(figsize=(7.0, 2.42))
    axs = [fig.add_axes([0.005, 0.09, 0.315, 0.86]),
           fig.add_axes([0.345, 0.09, 0.315, 0.86]),
           fig.add_axes([0.685, 0.09, 0.315, 0.86])]
    for ax in axs:
        panel_frame(ax)
    P = constellation()
    order = [0, 1, 2, 3, 4, 5, 0]

    # (a) 唯一到旋转
    ax = axs[0]
    ax.add_patch(Circle((0, 0), 0.84, facecolor="none", edgecolor=fs.BLUE,
                        lw=1.2, ls=(0, (4, 3)), zorder=2))
    th = np.deg2rad([22, 78, 136, 204, 262, 318])
    ax.scatter(0.84 * np.cos(th), 0.84 * np.sin(th), s=30, color=fs.BLUE,
               edgecolor="white", linewidth=0.7, zorder=4)
    ax.plot([0], [0], marker="+", ms=7, color=fs.FAINT, zorder=3)
    fs.label(ax, 0.84 * np.cos(np.deg2rad(78)) - 0.02, 0.84 * np.sin(np.deg2rad(78)) + 0.20,
             r"$Qh^\star$", fs=9.4, color=fs.BLUE, ha="left")
    fig.text(0.162, 0.035, "(a) 唯一到旋转，一条轨道", ha="center", va="center", fontsize=9.0)

    # (b) 无谱间隙
    ax = axs[1]
    ax.add_patch(Circle((0, 0), 0.84, facecolor=fs.ACCENT, alpha=0.12, edgecolor="none", zorder=1))
    ax.add_patch(Circle((0, 0), 0.84, facecolor="none", edgecolor=fs.ACCENT,
                        lw=1.0, ls=(0, (4, 3)), zorder=2))
    rng = np.random.default_rng(7)
    rr = 0.84 * np.sqrt(rng.uniform(0.05, 1.0, 24))
    tt = rng.uniform(0, 2 * np.pi, 24)
    ax.scatter(rr * np.cos(tt), rr * np.sin(tt), s=19, color=fs.ACCENT,
               edgecolor="white", linewidth=0.5, zorder=4, alpha=0.9)
    ax.plot([0], [0], marker="+", ms=7, color=fs.FAINT, zorder=3)
    fs.label(ax, 0, 1.02, "子空间可以滑动", fs=8.4, color=fs.ACCENT)
    fig.text(0.502, 0.035, "(b) 无谱间隙，一整片", ha="center", va="center", fontsize=9.0)

    # (c) 边缘不同
    ax = axs[2]
    S = np.array([[1.30, 0.0], [0.0, 0.72]]) * 1.05
    P = P * 1.05
    Qs = P @ S.T
    ax.plot(P[order, 0], P[order, 1], color=fs.BLUE, lw=0.9, alpha=0.45, zorder=2)
    ax.plot(Qs[order, 0], Qs[order, 1], color=fs.GREEN, lw=0.9, alpha=0.55,
            ls=(0, (3, 2)), zorder=2)
    ax.scatter(P[:, 0], P[:, 1], s=24, color=fs.BLUE, zorder=4,
               edgecolor="white", linewidth=0.6)
    ax.scatter(Qs[:, 0], Qs[:, 1], s=24, facecolor="white", edgecolor=fs.GREEN,
               linewidth=0.9, zorder=4)
    for i in (0, 3):
        ax.plot([P[i, 0], Qs[i, 0]], [P[i, 1], Qs[i, 1]], color=fs.FAINT,
                lw=0.7, ls=(0, (1, 2)), zorder=3)
    ax.plot([0], [0], marker="+", ms=7, color=fs.FAINT, zorder=3)
    fs.label(ax, 0, 1.02, "不成比例的缩放", fs=8.4, color=fs.GREEN)
    fig.text(0.842, 0.035, "(c) 边缘不同，两套坐标", ha="center", va="center", fontsize=9.0)
    return fs.save(fig, "theory", "ch03", "failure-modes")


if __name__ == "__main__":
    fig_rotation_orbit()
    fig_failure_modes()
