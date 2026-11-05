# -*- coding: utf-8 -*-
"""第 12 章插图：互近邻的定义、模型拼接。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs


def _neighbors(pts, i, k):
    d = np.linalg.norm(pts - pts[i], axis=1)
    order = np.argsort(d)
    return set(int(j) for j in order[1:k + 1])


def fig_mnn():
    """(a)(b) 两个表征空间里同一个样本的近邻集合；(c) 随机基线 k/N。"""
    n, k = 12, 3
    rng = np.random.default_rng(0)
    gx = np.array([0.15, 0.38, 0.62, 0.85])
    gy = np.array([0.17, 0.50, 0.83])
    base = np.array([[x, y] for y in gy for x in gx])
    PA = base + rng.uniform(-0.045, 0.045, size=base.shape)
    PB = base + rng.uniform(-0.045, 0.045, size=base.shape)

    # 选一个近邻集合部分重合的样本当例子（本布局下重合数恰为 2）
    pick = None
    for i in range(n):
        inter = _neighbors(PA, i, k) & _neighbors(PB, i, k)
        if len(inter) == 2:
            pick = i
            break
    if pick is None:
        pick = 0
    sa = _neighbors(PA, pick, k)
    sb = _neighbors(PB, pick, k)
    shared = sa & sb
    print("示例样本", pick + 1, "A 的近邻", sorted(x + 1 for x in sa),
          "B 的近邻", sorted(x + 1 for x in sb), "重合", len(shared))

    fig = plt.figure(figsize=(7.0, 3.35))
    axA = fig.add_axes([0.035, 0.10, 0.225, 0.72])
    axB = fig.add_axes([0.305, 0.10, 0.225, 0.72])
    axC = fig.add_axes([0.625, 0.165, 0.345, 0.665])

    for ax, P, S, ttl in ((axA, PA, sa, "(a) 模型 A 的表征"),
                          (axB, PB, sb, "(b) 模型 B 的表征")):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        fs.clean(ax, spines=())
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#FCFDFE",
                               edgecolor="#D5DBE1", lw=0.8, zorder=0))
        for j in range(n):
            if j == pick or j in S:
                continue
            ax.scatter(P[j, 0], P[j, 1], s=26, facecolor="white",
                       edgecolor=fs.FAINT, linewidth=0.9, zorder=3)
            fs.label(ax, P[j, 0] + 0.038, P[j, 1] + 0.048, str(j + 1),
                     fs=7.4, color=fs.FAINT, ha="left")
        for j in S:
            c = fs.ACCENT if j in shared else fs.BLUE
            ax.plot([P[pick, 0], P[j, 0]], [P[pick, 1], P[j, 1]],
                    color=c, lw=1.0, alpha=0.75, zorder=2)
            ax.scatter(P[j, 0], P[j, 1], s=62, color=c, zorder=4,
                       edgecolor="white", linewidth=1.1)
            fs.label(ax, P[j, 0] + 0.045, P[j, 1] - 0.052, str(j + 1),
                     fs=8.2, color=c, ha="left", weight="bold")
        ax.scatter(P[pick, 0], P[pick, 1], s=130, color=fs.INK, marker="*",
                   zorder=5, edgecolor="white", linewidth=0.9)
        fs.label(ax, P[pick, 0] + 0.012, P[pick, 1] - 0.105, "样本 i",
                 fs=8.4, color=fs.INK, ha="center")
        fs.label(ax, 0.5, 1.055, ttl, fs=9.0, color=fs.MUTED)
        names = "{" + ", ".join(str(j + 1) for j in sorted(S)) + "}"
        fs.label(ax, 0.5, -0.075, "$S$ = " + names, fs=8.6, color=fs.INK)

    fs.label(axB, 0.5, -0.185, "橙色近邻两侧都有，蓝色只出现在一侧", fs=7.8, color=fs.MUTED)
    fs.label(axA, 0.5, -0.185, "重合 %d / %d" % (len(shared), k), fs=8.6,
             color=fs.ACCENT, weight="bold")

    # (c) 随机基线
    N = 1000
    ks = np.arange(1, 251)
    axC.plot(ks, ks / N, color=fs.BLUE, lw=1.6, zorder=3)
    axC.axhline(0.16, color=fs.ACCENT, lw=1.1, ls=(0, (4, 3)), zorder=2)
    axC.scatter([10], [10 / N], s=34, color=fs.BLUE, zorder=4,
                edgecolor="white", linewidth=1.0)
    axC.annotate("k = 10 时为 0.01", xy=(10, 0.01), xytext=(52, 0.035),
                 fontsize=8.2, color=fs.BLUE,
                 arrowprops=dict(arrowstyle="-", color=fs.BLUE, lw=0.8))
    axC.annotate("跨模态对齐的最高读数 0.16", xy=(96, 0.161), xytext=(16, 0.198),
                 fontsize=8.2, color=fs.ACCENT, va="center",
                 arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8))
    axC.set_xlim(0, 250); axC.set_ylim(0, 0.27)
    axC.set_xlabel("近邻个数 $k$", fontsize=9)
    axC.set_ylabel("随机基线 $k/N$", fontsize=9)
    fs.clean(axC, grid=True, spines=("left", "bottom"))
    axC.text(0.0, 1.05, "(c) 两条近邻集合随机时的期望重合度", transform=axC.transAxes,
             fontsize=9.0, color=fs.MUTED, ha="left", va="bottom")
    return fs.save(fig, "ch12", "mutual-knn")


def fig_stitch():
    """A 的前半 + 线性层 h + B 的后半。"""
    fig = plt.figure(figsize=(7.0, 3.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    xa, wa = 0.055, 0.170
    xb, wb = 0.560, 0.170
    xh, wh = 0.320, 0.150

    fs.label(ax, xa + wa / 2, 0.895, "模型 A", fs=9.6, weight="bold", color=fs.BLUE)
    fs.label(ax, xb + wb / 2, 0.895, "模型 B", fs=9.6, weight="bold", color=fs.GREEN)

    # A 列
    fs.box(ax, xa, 0.075, wa, 0.085, "输入 $x$", fc="#FFFFFF", ec=fs.MUTED, fs=8.8, lw=1.0)
    fs.box(ax, xa, 0.200, wa, 0.115, "$f_1$", fc="#EEF3FE", ec=fs.BLUE, fs=10, lw=1.1)
    fs.box(ax, xa, 0.385, wa, 0.115, "$f_k$", fc="#EEF3FE", ec=fs.BLUE, fs=10, lw=1.1)
    fs.box(ax, xa, 0.560, wa, 0.215, "A 剩下的层\n$f_{k+1}\\ \\cdots\\ f_n$",
           fc="#FFFFFF", ec=fs.FAINT, tc=fs.FAINT, fs=8.6, lw=1.0)
    fs.label(ax, xa + wa + 0.015, 0.667, "丢弃", fs=8.0, color=fs.FAINT, ha="left")

    # B 列
    fs.box(ax, xb, 0.115, wb, 0.215, "B 前面的层\n$g_1\\ \\cdots\\ g_k$",
           fc="#FFFFFF", ec=fs.FAINT, tc=fs.FAINT, fs=8.6, lw=1.0)
    fs.label(ax, xb - 0.015, 0.222, "丢弃", fs=8.0, color=fs.FAINT, ha="right")
    fs.box(ax, xb, 0.500, wb, 0.115, "$g_{k+1}$", fc="#EAF6F1", ec=fs.GREEN, fs=10, lw=1.1)
    fs.box(ax, xb, 0.700, wb, 0.115, "$g_m$", fc="#EAF6F1", ec=fs.GREEN, fs=10, lw=1.1)

    # 输出
    fs.box(ax, 0.800, 0.700, 0.165, 0.115, "输出与评测", fc="#FFF7F0", ec=fs.ACCENT,
           fs=8.8, lw=1.1)
    fs.label(ax, 0.8825, 0.665, "看掉多少点", fs=7.8, color=fs.MUTED)

    # 拼接层
    fs.box(ax, xh, 0.432, wh, 0.136, "线性层 $h$", fc="#FFF7F0", ec=fs.ACCENT,
           fs=9.4, lw=1.3)
    fs.label(ax, xh + wh / 2, 0.395, "唯一被训练的参数", fs=8.0, color=fs.ACCENT)

    # 箭头
    fs.arrow(ax, (xa + wa / 2, 0.162), (xa + wa / 2, 0.197), color=fs.MUTED, lw=1.1)
    fs.arrow(ax, (xa + wa / 2, 0.317), (xa + wa / 2, 0.382), color=fs.MUTED, lw=1.1)
    fs.label(ax, xa + wa + 0.018, 0.349, "$\\vdots$", fs=11, color=fs.FAINT, ha="left")
    fs.arrow(ax, (xa + wa + 0.002, 0.4425), (xh - 0.004, 0.4425), color=fs.ACCENT, lw=1.4)
    fs.arrow(ax, (xh + wh + 0.002, 0.5575), (xb - 0.004, 0.5575), color=fs.ACCENT, lw=1.4)
    fs.arrow(ax, (xb + wb / 2, 0.617), (xb + wb / 2, 0.697), color=fs.MUTED, lw=1.1)
    fs.label(ax, xb + wb / 2 + 0.030, 0.657, "$\\vdots$", fs=11, color=fs.FAINT, ha="left")
    fs.arrow(ax, (xb + wb + 0.002, 0.7575), (0.796, 0.7575), color=fs.MUTED, lw=1.1)

    fs.label(ax, 0.5, 0.028,
             "拼接后 $F = f_1 \\circ \\cdots \\circ f_k \\circ h \\circ g_{k+1} \\circ \\cdots \\circ g_m$",
             fs=9.2, color=fs.INK)
    return fs.save(fig, "ch12", "stitching")


if __name__ == "__main__":
    fig_mnn()
    fig_stitch()
