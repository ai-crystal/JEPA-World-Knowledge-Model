# -*- coding: utf-8 -*-
"""第 4 章插图：三种塌缩状态、InfoNCE 的吸引与排斥。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_states():
    """同一坐标范围下的三种状态：正常、完全塌缩、维度塌缩。数据为脚本生成的示意数据。"""
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.95))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.175, top=0.905, wspace=0.28)

    centers = np.array([[-1.15, 0.90], [1.05, 1.10], [0.10, -1.30]])
    cols = [fs.BLUE, fs.GREEN, fs.ACCENT]
    n = 55
    u = np.array([1.0, 0.52])
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])

    # (a) 正常：三个类各自成团，两个方向都有伸展
    shear = np.array([[1.0, 0.22], [0.0, 0.92]])
    pts_normal = [centers[k] + rng.normal(0, 0.30, (n, 2)) @ shear.T for k in range(3)]

    # (b) 完全塌缩：所有点落在原点
    pts_full = [np.zeros((n, 2)) for _ in range(3)]

    # (c) 维度塌缩：三个类沿同一条直线排开，其余方向被压平
    pts_dim = []
    for k, t0 in enumerate((-1.80, 0.0, 1.80)):
        t = t0 + rng.normal(0, 0.26, n)
        pts_dim.append(np.outer(t, u) + np.outer(rng.normal(0, 0.048, n), v))

    titles = ["(a) 正常", "(b) 完全塌缩", "(c) 维度塌缩"]
    subs = ["两个方向都在变化", "所有点缩成一处", "只剩一个方向在变化"]

    for i, ax in enumerate(axes):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(length=2.6, labelsize=7.8, pad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(fs.FAINT)
        ax.set_title(titles[i], fontsize=9.6, color=fs.INK, pad=5)
        ax.set_xlabel(subs[i], fontsize=8.3, color=fs.MUTED, labelpad=4)

    # (a)
    for k in range(3):
        axes[0].scatter(pts_normal[k][:, 0], pts_normal[k][:, 1], s=11, color=cols[k],
                        alpha=0.85, linewidths=0, zorder=3)

    # (b)
    axes[1].scatter([0], [0], s=58, color=fs.INK, linewidths=0, zorder=4)
    axes[1].annotate("所有点重合", xy=(0.07, 0.07), xytext=(0.85, 1.45),
                     fontsize=8.2, color=fs.MUTED, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.9,
                                     shrinkA=1, shrinkB=3))

    # (c)
    axes[2].plot([-3.15 * u[0], 3.15 * u[0]], [-3.15 * u[1], 3.15 * u[1]],
                 color=fs.FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for k in range(3):
        axes[2].scatter(pts_dim[k][:, 0], pts_dim[k][:, 1], s=11, color=cols[k],
                        alpha=0.85, linewidths=0, zorder=3)

    return fs.save(fig, "basics", "ch04", "three-states")


def fig_infonce():
    """一个锚点、一个正样本、四个负样本的吸引与排斥。"""
    fig = plt.figure(figsize=(7.0, 3.3))
    ax = fig.add_axes([0.005, 0.02, 0.595, 0.96])
    ax.set_xlim(-1.50, 1.44)
    ax.set_ylim(-1.34, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")

    R = 0.75
    pol = lambda deg, r=R: r * np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])

    ax.add_patch(plt.Circle((0, 0), R, fill=False, ec=fs.FAINT, lw=0.9,
                            ls=(0, (4, 3)), zorder=1))

    anchor = pol(172)
    pos = pol(125)
    negs = [pol(a) for a in (45, 350, 300, 230)]

    # 负样本的排斥箭头：从点向外
    for p in negs:
        fs.arrow(ax, tuple(p), tuple(p * 1.26), color=fs.FAINT, lw=1.0,
                 ls=(0, (3, 2)), ms=7, z=2)

    # 正样本的吸引箭头
    d = (pos - anchor) / np.linalg.norm(pos - anchor)
    fs.arrow(ax, tuple(anchor + d * 0.075), tuple(pos - d * 0.085),
             color=fs.ACCENT, lw=1.8, ms=11, z=4)

    ax.scatter(*anchor, s=52, color=fs.INK, zorder=6, linewidths=0)
    ax.scatter(*pos, s=52, color=fs.ACCENT, zorder=6, linewidths=0)
    for p in negs:
        ax.scatter(*p, s=40, color=fs.FAINT, zorder=6, linewidths=0)

    fs.label(ax, -1.26, 0.09, "锚点", fs=9.2, color=fs.INK, weight="bold", ha="left")
    fs.label(ax, -0.42, 0.87, "正样本", fs=9.2, color=fs.ACCENT, weight="bold", ha="center")
    fs.label(ax, -0.33, 0.22, "拉近", fs=9.0, color=fs.ACCENT, ha="center")
    fs.label(ax, 0.80, 0.47, "负样本", fs=9.2, color=fs.MUTED, weight="bold", ha="left")
    fs.label(ax, 0.66, -0.93, "推远", fs=9.0, color=fs.MUTED, ha="left")

    # 右侧说明
    axt = fig.add_axes([0.630, 0.04, 0.360, 0.92])
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.axis("off")
    items = [
        (0.86, "锚点", fs.INK, "bold", 9.4),
        (0.775, "被比较的那个视图", fs.MUTED, "normal", 8.4),
        (0.63, "正样本", fs.ACCENT, "bold", 9.4),
        (0.545, "同一物体的另一个视图", fs.MUTED, "normal", 8.4),
        (0.40, "四个负样本", fs.MUTED, "bold", 9.4),
        (0.315, "同一批里的其他物体", fs.MUTED, "normal", 8.4),
        (0.19, "全部先归一化到球面", fs.FAINT, "normal", 8.4),
    ]
    for y, txt, c, w, s in items:
        fs.label(axt, 0.0, y, txt, fs=s, color=c, weight=w, ha="left")

    return fs.save(fig, "basics", "ch04", "infonce")


if __name__ == "__main__":
    fig_states()
    fig_infonce()
