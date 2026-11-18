# -*- coding: utf-8 -*-
"""第 8 章插图：四种正样本、nuisance 分解、改一个旋钮的相对影响。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import figstyle as fs


# ---------------------------------------------------------------- 图 8-1
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
    return fs.save(fig, "practice", "ch08", "four-positives")


# ---------------------------------------------------------------- 图 8-2
def fig_nuisance():
    fig = plt.figure(figsize=(6.9, 3.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fs.label(ax, 0.235, 0.955, "一个场景，两次观测", fs=8.6, color=fs.MUTED)
    fs.label(ax, 0.735, 0.955, "白化后按方向切开", fs=8.6, color=fs.MUTED)

    # 生成过程
    fs.box(ax, 0.170, 0.470, 0.130, 0.105, "场景 $z$", fc="#EEF2FF", ec=fs.BLUE, fs=9.6, lw=1.15)
    fs.box(ax, 0.028, 0.790, 0.150, 0.100, "干扰 $\\nu_1$", fc="#FFF7ED", ec=fs.ACCENT, fs=9.2, lw=1.15)
    fs.box(ax, 0.245, 0.790, 0.145, 0.100, "观测 $x_1$", fc="#FFFFFF", ec=fs.INK, fs=9.6, lw=1.15)
    fs.box(ax, 0.028, 0.160, 0.150, 0.100, "干扰 $\\nu_2$", fc="#FFF7ED", ec=fs.ACCENT, fs=9.2, lw=1.15)
    fs.box(ax, 0.245, 0.160, 0.145, 0.100, "观测 $x_2$", fc="#FFFFFF", ec=fs.INK, fs=9.6, lw=1.15)

    fs.arrow(ax, (0.180, 0.845), (0.242, 0.845), color=fs.ACCENT, lw=1.1)
    fs.arrow(ax, (0.180, 0.210), (0.242, 0.210), color=fs.ACCENT, lw=1.1)
    fs.arrow(ax, (0.240, 0.572), (0.300, 0.786), color=fs.INK, lw=1.1, rad=-0.10)
    fs.arrow(ax, (0.240, 0.472), (0.300, 0.264), color=fs.INK, lw=1.1, rad=0.10)
    fs.label(ax, 0.238, 0.722, "$g_1$", fs=8.6, color=fs.MUTED)
    fs.label(ax, 0.238, 0.318, "$g_2$", fs=8.6, color=fs.MUTED)
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
    return fs.save(fig, "practice", "ch08", "nuisance")


# ---------------------------------------------------------------- 图 8-3
def fig_knobs():
    """表 8 摘录：一次只改一个旋钮，相对默认设置的 R@1 变化。"""
    base = .246
    rows = [
        ("整句当视图", .185, "trim"),
        ("温和修剪", .202, "trim"),
        ("λ=0.1，加裁剪", .202, "other"),
        ("BERT-T，192 宽", .234, "other"),
        ("batch 512", .241, "other"),
        ("不掩码", .242, "other"),
        ("4+4 视图", .246, "other"),
        ("默认，2+6 视图", .246, "base"),
        ("2+10 视图", .248, "other"),
        ("掩码 30%", .254, "other"),
        ("lr 2e−4", .255, "other"),
        ("BERT-B，768 宽", .260, "other"),
        ("2+2 视图，加裁剪", .262, "other"),
    ]
    n = len(rows)
    fig = plt.figure(figsize=(6.9, 4.35))
    ax = fig.add_axes([0.295, 0.155, 0.615, 0.775])

    y = np.arange(n)
    ax.axvspan((.230 / base - 1) * 100, (.260 / base - 1) * 100,
               color=fs.LIGHT, zorder=0)
    ax.axvline(0, color=fs.INK, lw=1.0, zorder=2)

    for i, (name, val, kind) in enumerate(rows):
        d = (val / base - 1) * 100
        if kind == "base":
            continue
        col = fs.ACCENT if kind == "trim" else fs.BLUE
        ax.barh(i, d, height=0.60, color=col, zorder=3,
                alpha=1.0 if kind == "trim" else 0.75, edgecolor="none")
        tx = d - 0.9 if d < 0 else d + 0.9
        ha = "right" if d < 0 else "left"
        ax.text(tx, i, ("%.0f%%" % d).replace("-", "−"), va="center", ha=ha, fontsize=7.8,
                color=fs.ACCENT if kind == "trim" else fs.MUTED,
                weight="bold" if kind == "trim" else "normal", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(["%s  %.3f" % (r[0], r[1]) for r in rows], fontsize=8.6)
    for lab, r in zip(ax.get_yticklabels(), rows):
        if r[2] == "trim":
            lab.set_color(fs.ACCENT)
            lab.set_fontweight("bold")
        elif r[2] == "base":
            lab.set_color(fs.INK)
            lab.set_fontweight("bold")
        else:
            lab.set_color(fs.INK)
    ax.set_ylim(-0.75, n - 0.25)
    ax.set_xlim(-31, 13)
    ax.set_xticks([-30, -20, -10, 0, 10])
    ax.set_xticklabels(["−30", "−20", "−10", "0", "+10"])
    ax.set_xlabel("相对默认设置的 R@1 变化（%），DINOv2 冻结评分", fontsize=9)
    fs.clean(ax, spines=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_color(fs.FAINT)

    ax.text(0.0, -0.175, "灰带：默认设置三个种子的散布（.230–.260）",
            transform=ax.transAxes, fontsize=7.6, color=fs.MUTED, ha="left")
    ax.text(0.0, -0.235, "另有五个设置整轮塌缩，未列入本图",
            transform=ax.transAxes, fontsize=7.6, color=fs.MUTED, ha="left")
    return fs.save(fig, "practice", "ch08", "knobs")


if __name__ == "__main__":
    fig_views()
    fig_nuisance()
    fig_knobs()
