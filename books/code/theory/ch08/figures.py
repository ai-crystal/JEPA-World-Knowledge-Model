# -*- coding: utf-8 -*-
"""第 8 章插图：三种等价类、识别性阶梯、白化前后的可行解集、可辨识性结论的对照。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import figstyle as fs


# ---------------------------------------------------------------- 公共
def polygon(n=5, r=0.78, off=90.0):
    th = np.deg2rad(np.array([off + 360.0 * i / n for i in range(n)]))
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


PTS = polygon(4, 0.78, 108.0)


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def draw_shape(ax, X, color, alpha=1.0, lw=1.0, ms=20, z=3):
    loop = np.vstack([X, X[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color=color, lw=lw, alpha=alpha, zorder=z)
    ax.scatter(X[:, 0], X[:, 1], s=ms, color=color, alpha=alpha, zorder=z + 1,
               edgecolor="white", linewidth=0.6)


def frame(ax, lim=1.22):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=fs.GRID, lw=0.9,
                        ls=(0, (3, 3)), zorder=1))


def ellipse_of(G, n=240):
    t = np.linspace(0, 2 * np.pi, n)
    C = np.stack([np.cos(t), np.sin(t)], axis=1)
    return C @ G.T


# ---------------------------------------------------------------- 图 8-1
def fig_equivalence():
    fig = plt.figure(figsize=(7.0, 2.72))

    # (a) 可逆线性：形状被拉伸剪切
    axa = fig.add_axes([0.010, 0.245, 0.300, 0.690])
    frame(axa, lim=1.34)
    draw_shape(axa, PTS, fs.FAINT, alpha=0.75, lw=0.9, ms=13, z=2)
    for M, c in [(np.array([[1.04, 0.34], [-0.08, 0.76]]), fs.BLUE),
                 (np.array([[0.78, -0.30], [0.34, 1.00]]), fs.ACCENT)]:
        E = ellipse_of(M)
        axa.plot(E[:, 0], E[:, 1], color=c, lw=1.0, alpha=0.45, zorder=2)
        draw_shape(axa, PTS @ M.T, c, alpha=0.85, lw=1.1, ms=15)
    fs.label(axa, 0.0, -1.224, "拉伸与剪切", fs=8.6, color=fs.MUTED)
    fs.label(axa, 0.0, 1.208, "(a) $GL(k)$", fs=10.2, color=fs.BLUE)

    # (b) 正交：只转角度
    axb = fig.add_axes([0.325, 0.245, 0.300, 0.690])
    frame(axb, lim=1.34)
    draw_shape(axb, PTS, fs.FAINT, alpha=0.75, lw=0.9, ms=13, z=2)
    R = rot(np.deg2rad(25))
    draw_shape(axb, PTS @ R.T, fs.GREEN, alpha=0.90, lw=1.1, ms=15)
    p0 = PTS[3]
    p1 = p0 @ R.T
    axb.annotate("", xy=p1, xytext=p0,
                 arrowprops=dict(arrowstyle="-|>", color=fs.GREEN, lw=1.0,
                                 mutation_scale=8, connectionstyle="arc3,rad=0.28"))
    axb.add_patch(Circle((0, 0), 0.78, fill=False, ec=fs.INK, lw=0.9, alpha=0.40,
                         ls=(0, (2, 2)), zorder=1))
    fs.label(axb, 0.0, -1.224, "只改朝向", fs=8.6, color=fs.MUTED)
    fs.label(axb, 0.0, 1.208, "(b) $O(k)$", fs=10.2, color=fs.GREEN)

    # (c) 置换：只改名字
    axc = fig.add_axes([0.650, 0.135, 0.345, 0.795])
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")
    fs.label(axc, 0.50, 0.955, "(c) 置换", fs=10.2, color=fs.PURPLE)
    order_a = [1, 2, 3]
    order_b = [3, 1, 2]
    colmap = {1: fs.BLUE, 2: fs.ACCENT, 3: fs.GREEN}
    for i, lab in enumerate(order_a):
        y = 0.715 - i * 0.175
        fs.box(axc, 0.020, y - 0.062, 0.245, 0.124, "成分 %d" % lab, fc="white",
               ec=colmap[lab], fs=8.8, lw=1.0)
    for i, lab in enumerate(order_b):
        y = 0.715 - i * 0.175
        fs.box(axc, 0.735, y - 0.062, 0.245, 0.124, "成分 %d" % lab, fc="white",
               ec=colmap[lab], fs=8.8, lw=1.0)
    fs.arrow(axc, (0.290, 0.540), (0.710, 0.540), color=fs.PURPLE, lw=1.2)
    fs.label(axc, 0.500, 0.610, "$P$", fs=11.0, color=fs.PURPLE)
    fs.label(axc, 0.500, 0.440, "三个成分换了位置", fs=8.2, color=fs.MUTED)
    fs.label(axc, 0.500, 0.085, "$P$ 也是正交矩阵，落在 $O(k)$ 里", fs=8.2, color=fs.MUTED)

    fs.save(fig, "theory", "ch08", "equivalence")


# ---------------------------------------------------------------- 图 8-2
def fig_ladder():
    fig = plt.figure(figsize=(6.9, 2.95))
    ax = fig.add_axes([0.030, 0.060, 0.950, 0.880])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        (0.095, 0.370, 0.770, "$GL(k)$", "$k^2$ 个参数", "只有正样本", fs.BLUE),
        (0.370, 0.660, 0.515, "$O(k)$", "$k(k-1)/2$ 个参数", "平方距离加白化", fs.GREEN),
        (0.660, 0.945, 0.260, "置换", "$k!$ 个元素，维数 0", "独立性加非高斯", fs.PURPLE),
    ]
    for x0, x1, y, name, cnt, hypo, col in steps:
        ax.plot([x0, x1], [y, y], color=col, lw=2.6, solid_capstyle="butt", zorder=3)
        ax.plot([x0, x0], [y - 0.105, y], color=col, lw=0.9, ls=(0, (2, 2)), zorder=2)
        fs.label(ax, (x0 + x1) / 2, y + 0.098, name, fs=11.0, color=col)
        fs.label(ax, (x0 + x1) / 2, y + 0.045, cnt, fs=8.8, color=fs.INK)
        fs.label(ax, (x0 + x1) / 2, y - 0.058, hypo, fs=8.4, color=fs.MUTED)

    ax.plot([0.370, 0.370], [0.515, 0.770], color=fs.FAINT, lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.plot([0.660, 0.660], [0.260, 0.515], color=fs.FAINT, lw=0.9, ls=(0, (2, 2)), zorder=1)

    fs.arrow(ax, (0.048, 0.825), (0.048, 0.215), color=fs.FAINT, lw=1.1, ms=9)
    fs.label(ax, 0.022, 0.520, "可辨识性增强", fs=8.4, color=fs.MUTED, rotation=90)

    fs.label(ax, 0.095, 0.955, "每加一层统计结构，轨道就小一圈", fs=8.6, color=fs.MUTED, ha="left")
    fs.label(ax, 0.095, 0.052, "约束越来越强，能说的话越来越具体", fs=8.2,
             color=fs.MUTED, ha="left")

    fs.save(fig, "theory", "ch08", "ladder")


# ---------------------------------------------------------------- 图 8-3
def fig_whitening():
    fam = []
    for th, sa, sb in [(0, 1.46, 0.72), (50, 1.50, 0.60), (100, 1.40, 0.82),
                       (145, 1.48, 0.74)]:
        R = rot(np.deg2rad(th))
        S = np.array([[sa, 0.0], [0.0, sb]])
        fam.append(R @ S @ R.T)

    fig = plt.figure(figsize=(6.7, 3.00))

    axa = fig.add_axes([0.020, 0.185, 0.440, 0.695])
    frame(axa, lim=1.62)
    for G in fam:
        E = ellipse_of(G)
        axa.plot(E[:, 0], E[:, 1], color=fs.FAINT, lw=0.8, alpha=0.55, zorder=2)
    E = ellipse_of(fam[0])
    axa.plot(E[:, 0], E[:, 1], color=fs.BLUE, lw=1.3, alpha=0.95, zorder=3)
    fs.label(axa, 0.0, 1.505, "(a) 白化之前", fs=9.8, color=fs.BLUE)
    fs.label(axa, 0.0, -1.545, "每种椭圆形状都对应一个最优解", fs=8.4, color=fs.MUTED)

    axb = fig.add_axes([0.540, 0.185, 0.440, 0.695])
    frame(axb, lim=1.62)
    axb.add_patch(Circle((0, 0), 1.0, fill=False, ec=fs.GREEN, lw=1.6, zorder=3))
    for a in (0, 62):
        axb.plot([0, np.cos(np.deg2rad(a))], [0, np.sin(np.deg2rad(a))],
                 color=fs.GREEN, lw=0.9, alpha=0.55, ls=(0, (2, 2)), zorder=2)
    t = np.deg2rad(np.linspace(0, 62, 40))
    axb.plot(0.45 * np.cos(t), 0.45 * np.sin(t), color=fs.GREEN, lw=0.8, alpha=0.7,
             zorder=2)
    axb.annotate("", xy=(0.45 * np.cos(np.deg2rad(62)), 0.45 * np.sin(np.deg2rad(62))),
                 xytext=(0.45 * np.cos(np.deg2rad(52)), 0.45 * np.sin(np.deg2rad(52))),
                 arrowprops=dict(arrowstyle="-|>", color=fs.GREEN, lw=0.9,
                                 mutation_scale=7))
    fs.label(axb, 0.30, 0.30, r"$\theta$", fs=9.6, color=fs.GREEN)
    fs.label(axb, 0.0, 1.505, "(b) 白化之后", fs=9.8, color=fs.GREEN)
    fs.label(axb, 0.0, -1.545, "只剩旋转，而圆转过去还是圆", fs=8.4, color=fs.MUTED)

    fs.save(fig, "theory", "ch08", "whitening")


# ---------------------------------------------------------------- 图 8-4
def fig_identifiability():
    fig = plt.figure(figsize=(7.0, 3.30))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fs.label(ax, 0.030, 0.950, "生成过程的假设", fs=9.2, color=fs.MUTED, ha="left")
    fs.label(ax, 0.985, 0.950, "对最优表征的结论", fs=9.2, color=fs.MUTED, ha="right")

    hyps = [
        "内容与风格独立\n$p(c,s)=p(c)\\,p(s)$",
        "生成过程单射\n$g$ 不把两个状态粘在一起",
        "增强只作用于风格\n$g(c,a(s))=T_a\\,g(c,s)$",
        "增强覆盖每个观测的邻域\n$T_a$ 局部可逆，正样本足够密",
    ]
    ys = [0.785, 0.585, 0.385, 0.185]
    for h, y in zip(hyps, ys):
        fs.box(ax, 0.030, y - 0.070, 0.395, 0.140, h, fc="#F8FAFC", ec=fs.BLUE,
               fs=8.8, lw=1.0)

    cons = [
        "表征只依赖内容\n风格方向被丢弃",
        "内容只定到可逆线性变换\n不是旋转，也不是逐点",
    ]
    cys = [0.685, 0.385]
    for c, y in zip(cons, cys):
        fs.box(ax, 0.570, y - 0.082, 0.395, 0.164, c, fc="#FFFDF7", ec=fs.ACCENT,
               fs=9.2, lw=1.1)

    ax.plot([0.455, 0.455], [0.125, 0.845], color=fs.FAINT, lw=1.1, zorder=1)
    ax.plot([0.455, 0.478], [0.125, 0.125], color=fs.FAINT, lw=1.1, zorder=1)
    ax.plot([0.455, 0.478], [0.845, 0.845], color=fs.FAINT, lw=1.1, zorder=1)
    fs.arrow(ax, (0.472, 0.685), (0.565, 0.685), color=fs.INK, lw=1.2, ms=9)
    fs.arrow(ax, (0.472, 0.385), (0.565, 0.385), color=fs.INK, lw=1.2, ms=9)
    fs.label(ax, 0.512, 0.622, "总体最优", fs=7.6, color=fs.MUTED)
    fs.label(ax, 0.512, 0.322, "无穷多增强", fs=7.6, color=fs.MUTED)

    fs.label(ax, 0.030, 0.052, "示意图，条件与结论按 von Kügelgen 等 2021 的陈述概括，不含数值。",
             fs=8.0, color=fs.MUTED, ha="left")

    fs.save(fig, "theory", "ch08", "identifiability")


if __name__ == "__main__":
    fig_equivalence()
    fig_ladder()
    fig_whitening()
    fig_identifiability()
