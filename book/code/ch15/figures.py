# -*- coding: utf-8 -*-
"""第 15 章插图：四种正样本、nuisance 分解、下一词预测的 JEPA 形式。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import figstyle as fs


# ---------------------------------------------------------------- 图 15-1
def _cluster(cx, cy, dx=0.024, dy=0.072):
    """一个场景的四个观测，2x2 摆开，返回坐标列表。"""
    return [(cx - dx, cy + dy), (cx + dx, cy + dy),
            (cx - dx, cy - dy), (cx + dx, cy - dy)]


def _within(ax, pts, color=fs.BLUE, lw=1.15, z=2):
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=color, lw=lw, zorder=z, solid_capstyle="round")


def _selfloop(ax, pts, color=fs.BLUE):
    for p in pts:
        ax.add_patch(Circle(p, 0.0125, fill=False, ls=(0, (2, 2)), lw=0.95,
                            edgecolor=color, zorder=2))


def _link(ax, p, q, color=fs.BLUE, pad=0.016, **kw):
    """从点边缘出发的连线，避开点与自配环。"""
    p = np.asarray(p, float); q = np.asarray(q, float)
    d = q - p
    d = d / np.linalg.norm(d)
    # 纵向与横向的单位长度对应不同英寸，按坐标尺度归一
    fs.arrow(ax, tuple(p + d * pad * np.array([0.75, 1.0])),
             tuple(q - d * pad * np.array([0.75, 1.0])), color=color, **kw)


def fig_views():
    fig = plt.figure(figsize=(7.0, 3.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    cxs = [0.135, 0.375, 0.615, 0.855]
    cy = 0.675
    off = 0.052
    heads = ["(a) 好视图", "(b) 坏视图，弱", "(c) 坏视图，强", "(d) 下一词预测"]
    notes = [("同一场景连成一片", "跨场景一条也不连"),
             ("正样本多是自己", "要靠数据碰出重叠"),
             ("跨场景也撞在一起", "核被永久污染"),
             ("前缀接自己的续写", "高频词还往外漏")]

    A, B = [], []
    for cx in cxs:
        A.append(_cluster(cx - off, cy))
        B.append(_cluster(cx + off, cy))

    # (a) 场景内全连通
    _within(ax, A[0]); _within(ax, B[0])

    # (b) 自己配自己 + 偶尔碰到另一句
    _selfloop(ax, A[1]); _selfloop(ax, B[1])
    _link(ax, A[1][0], A[1][3], pad=0.019, style="-")
    _link(ax, B[1][1], B[1][2], pad=0.019, style="-")

    # (c) 自己配自己 + 跨场景相连
    _selfloop(ax, A[2]); _selfloop(ax, B[2])
    _link(ax, A[2][0], B[2][3], color=fs.ACCENT, pad=0.021, style="-")
    _link(ax, A[2][3], B[2][0], color=fs.ACCENT, pad=0.021, style="-")

    # (d) 前缀 → 续写（同一场景内），高频词外溢
    for grp in (A[3], B[3]):
        _link(ax, grp[0], grp[1], pad=0.014, ms=7)
        _link(ax, grp[2], grp[3], pad=0.014, ms=7)
    _link(ax, A[3][1], B[3][2], color=fs.ACCENT, pad=0.021, rad=-0.22, ms=7,
          ls=(0, (3.2, 2.0)))

    # 圆点、面板标题、结论文本
    for k, cx in enumerate(cxs):
        for p in A[k] + B[k]:
            ax.scatter([p[0]], [p[1]], s=27, color=fs.INK, zorder=4,
                       edgecolor="white", linewidth=0.7)
        fs.label(ax, cx, 0.930, heads[k], fs=9.3, weight="bold")
        fs.label(ax, cx - off, 0.532, "场景甲", fs=7.6, color=fs.MUTED)
        fs.label(ax, cx + off, 0.532, "场景乙", fs=7.6, color=fs.MUTED)
        fs.label(ax, cx, 0.408, notes[k][0], fs=7.8, color=fs.INK)
        fs.label(ax, cx, 0.318, notes[k][1], fs=7.8, color=fs.ACCENT)

    # 图例
    y = 0.135
    ax.plot([0.075, 0.125], [y, y], color=fs.BLUE, lw=1.3, solid_capstyle="round")
    fs.label(ax, 0.133, y, "连同一场景的两次观测", fs=7.6, color=fs.MUTED, ha="left")
    ax.plot([0.360, 0.410], [y, y], color=fs.ACCENT, lw=1.3, solid_capstyle="round")
    fs.label(ax, 0.418, y, "连到了另一个场景", fs=7.6, color=fs.MUTED, ha="left")
    ax.add_patch(Circle((0.632, y), 0.013, fill=False, ls=(0, (2, 2)), lw=1.0,
                        edgecolor=fs.BLUE, zorder=2))
    fs.label(ax, 0.652, y, "观测与自己配对", fs=7.6, color=fs.MUTED, ha="left")
    return fs.save(fig, "ch15", "four-positives")


# ---------------------------------------------------------------- 图 15-2
def fig_nuisance():
    fig = plt.figure(figsize=(6.9, 3.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fs.label(ax, 0.235, 0.955, "一个场景，两次观测", fs=8.6, color=fs.MUTED)
    fs.label(ax, 0.735, 0.955, "白化后按方向切开", fs=8.6, color=fs.MUTED)

    # 生成过程
    fs.box(ax, 0.170, 0.470, 0.130, 0.105, "场景 $z$", fc="#EEF2FF", ec=fs.BLUE, fs=9.6, lw=1.15)
    fs.box(ax, 0.035, 0.790, 0.135, 0.100, "干扰 $\\nu_1$", fc="#FFF7ED", ec=fs.ACCENT, fs=9.6, lw=1.15)
    fs.box(ax, 0.245, 0.790, 0.145, 0.100, "观测 $x_1$", fc="#FFFFFF", ec=fs.INK, fs=9.6, lw=1.15)
    fs.box(ax, 0.035, 0.160, 0.135, 0.100, "干扰 $\\nu_2$", fc="#FFF7ED", ec=fs.ACCENT, fs=9.6, lw=1.15)
    fs.box(ax, 0.245, 0.160, 0.145, 0.100, "观测 $x_2$", fc="#FFFFFF", ec=fs.INK, fs=9.6, lw=1.15)

    fs.arrow(ax, (0.172, 0.845), (0.242, 0.845), color=fs.ACCENT, lw=1.1)
    fs.arrow(ax, (0.172, 0.210), (0.242, 0.210), color=fs.ACCENT, lw=1.1)
    fs.arrow(ax, (0.240, 0.572), (0.300, 0.786), color=fs.INK, lw=1.1, rad=-0.10)
    fs.arrow(ax, (0.240, 0.472), (0.300, 0.264), color=fs.INK, lw=1.1, rad=0.10)
    fs.label(ax, 0.243, 0.722, "$g_1$", fs=8.6, color=fs.MUTED)
    fs.label(ax, 0.243, 0.318, "$g_2$", fs=8.6, color=fs.MUTED)
    fs.label(ax, 0.100, 0.432, "$x = g(z,\\nu)$", fs=9.0, color=fs.MUTED)

    # 两个嵌入的 k 个方向
    n, x0, cw = 10, 0.560, 0.036
    for row, (yy, nblue) in enumerate(((0.755, 6), (0.285, 6))):
        for i in range(n):
            blue = i < nblue
            ax.add_patch(plt.Rectangle((x0 + i * cw, yy), cw - 0.004, 0.112,
                                       facecolor="#DBEAFE" if blue else "#F3F4F6",
                                       edgecolor=fs.BLUE if blue else fs.FAINT,
                                       lw=0.8, zorder=3))
    fs.label(ax, x0, 0.895, "编码 $f_1(x_1)$ 的坐标方向", fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax, x0, 0.232, "编码 $f_2(x_2)$ 的坐标方向", fs=8.2, color=fs.MUTED, ha="left")

    # 蓝格：能对上
    for i in range(6):
        xc = x0 + i * cw + (cw - 0.004) / 2
        ax.annotate("", xy=(xc, 0.762), xytext=(xc, 0.400),
                    arrowprops=dict(arrowstyle="<|-|>", color=fs.BLUE, lw=0.85,
                                    mutation_scale=5.5))
    # 灰格：对不上
    for i in (7, 9):
        xc = x0 + i * cw + (cw - 0.004) / 2
        ax.plot([xc, xc], [0.700, 0.762], color=fs.ACCENT, lw=0.9, ls=(0, (2, 2)), zorder=2)
        ax.plot([xc, xc], [0.400, 0.462], color=fs.ACCENT, lw=0.9, ls=(0, (2, 2)), zorder=2)
    xc = x0 + 8 * cw + (cw - 0.004) / 2
    fs.label(ax, xc, 0.575, "$\\times$", fs=15, color=fs.ACCENT)
    fs.label(ax, xc, 0.645, "旋转搬不动", fs=7.6, color=fs.ACCENT)

    # 小图例
    y = 0.095
    ax.add_patch(plt.Rectangle((x0, y - 0.022), 0.030, 0.044, facecolor="#DBEAFE",
                               edgecolor=fs.BLUE, lw=0.8, zorder=3))
    fs.label(ax, x0 + 0.040, y, "承载场景的方向", fs=7.8, color=fs.MUTED, ha="left")
    ax.add_patch(plt.Rectangle((x0 + 0.180, y - 0.022), 0.030, 0.044, facecolor="#F3F4F6",
                               edgecolor=fs.FAINT, lw=0.8, zorder=3))
    fs.label(ax, x0 + 0.220, y, "被干扰占用的方向", fs=7.8, color=fs.MUTED, ha="left")
    return fs.save(fig, "ch15", "nuisance")


# ---------------------------------------------------------------- 图 15-3
def fig_next_token_jepa():
    fig = plt.figure(figsize=(6.9, 3.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0.11, 0.93); ax.axis("off")

    fs.label(ax, 0.020, 0.895, "正样本取自同一序列之内", fs=8.2, color=fs.MUTED, ha="left")

    fs.box(ax, 0.015, 0.565, 0.148, 0.160, "前缀 $c$", fc="#EEF2FF", ec=fs.BLUE, fs=10, lw=1.15)
    fs.label(ax, 0.089, 0.528, "前 $t$ 个词", fs=7.8, color=fs.MUTED)

    fs.box(ax, 0.200, 0.565, 0.150, 0.160, "编码器 $h$", fc="#FFFFFF", ec=fs.INK, fs=9.6, lw=1.15)
    fs.arrow(ax, (0.165, 0.645), (0.198, 0.645), color=fs.INK, lw=1.15)

    fs.box(ax, 0.390, 0.578, 0.112, 0.135, "$h(c)$", fc="#FFFFFF", ec=fs.INK, fs=9.6, lw=1.15)
    fs.label(ax, 0.446, 0.540, "上下文向量", fs=7.6, color=fs.MUTED)
    fs.arrow(ax, (0.352, 0.645), (0.388, 0.645), color=fs.INK, lw=1.15)

    fs.box(ax, 0.598, 0.578, 0.130, 0.135, "$\\langle h, w_v\\rangle$", fc="#F5F3FF",
           ec=fs.PURPLE, fs=10, lw=1.15)
    fs.arrow(ax, (0.504, 0.645), (0.596, 0.645), color=fs.INK, lw=1.15)

    fs.box(ax, 0.390, 0.190, 0.152, 0.135, "词向量表 $\\{w_v\\}$", fc="#FFF7ED",
           ec=fs.ACCENT, fs=9.4, lw=1.15)
    fs.label(ax, 0.466, 0.157, "每个词一行", fs=7.6, color=fs.MUTED)
    fs.arrow(ax, (0.544, 0.262), (0.648, 0.572), color=fs.ACCENT, lw=1.15, rad=-0.14)

    # 预测项
    fs.box(ax, 0.775, 0.645, 0.205, 0.150, "$-\\langle h, w_y\\rangle$\n预测项",
           fc="#FFFFFF", ec=fs.BLUE, fs=9.6, lw=1.15)
    fs.arrow(ax, (0.730, 0.678), (0.773, 0.716), color=fs.BLUE, lw=1.15)

    # 反塌缩项
    fs.box(ax, 0.775, 0.330, 0.205, 0.165,
           "$\\log\\sum_v e^{\\langle h, w_v\\rangle}$\n反塌缩项 $A(h)$",
           fc="#FFFFFF", ec=fs.PURPLE, fs=9.6, lw=1.15)
    fs.arrow(ax, (0.730, 0.612), (0.773, 0.462), color=fs.PURPLE, lw=1.15, rad=0.12)

    # 两项相加得到损失：右侧汇流线 + 求和号
    ax.set_xlim(0, 1.13)
    ax.plot([0.980, 0.997], [0.720, 0.720], color=fs.INK, lw=1.0, zorder=2)
    ax.plot([0.980, 0.997], [0.4125, 0.4125], color=fs.INK, lw=1.0, zorder=2)
    ax.plot([0.997, 0.997], [0.4125, 0.720], color=fs.INK, lw=1.0, zorder=2)
    ax.add_patch(plt.Circle((0.997, 0.566), 0.026, facecolor="white", edgecolor=fs.INK,
                            lw=1.0, zorder=3))
    fs.label(ax, 0.997, 0.566, "$+$", fs=11, color=fs.INK)
    fs.arrow(ax, (1.024, 0.566), (1.070, 0.566), color=fs.INK, lw=1.1, ms=7)
    fs.label(ax, 1.078, 0.566, "损失", fs=9.2, color=fs.INK, ha="left")
    return fs.save(fig, "ch15", "next-token-jepa")


if __name__ == "__main__":
    fig_views()
    fig_nuisance()
    fig_next_token_jepa()
