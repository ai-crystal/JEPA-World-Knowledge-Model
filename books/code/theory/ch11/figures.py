# -*- coding: utf-8 -*-
"""第 11 章插图：证据链总览、三条硬限制的示意、限制与定理前提的对应矩阵。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import figstyle as fs


def fig_chain():
    """图 8-1：数据 → 视图 → 对齐 → 工具，四段流程与各自的读数。"""
    fig, ax = fs.canvas(7.0, 3.55)
    ax.set_xlim(0, 1); ax.set_ylim(0.035, 0.965)

    rows = [
        ("1　数据", fs.INK,
         "Flickr30k 两万九千张图，每张五条人工描述\n网页侧是九亿四千万张图片的文字",
         "145k 条 → 9.4 亿张", ""),
        ("2　视图", fs.ACCENT,
         "决定哪些观测算同一个场景的两次观测\n同图两段描述／同段文字两次损坏／前缀与续写",
         "定理 1 到 4", "好／坏／下一词"),
        ("3　对齐", fs.BLUE,
         "互 10-NN 与单旋转检索，两侧从不共享损失\nFlickr 上 .062 与 .047 与 .042",
         "42 倍", ".184 对 .154"),
        ("4　工具", fs.GREEN,
         "只用二十二万个参数的旋转当接口\n没有配对时靠点云几何把旋转找回来",
         "CIDEr .430", "零配对 16.4%"),
    ]

    ys = [0.845, 0.615, 0.385, 0.155]
    h = 0.132

    for i, ((name, col, body, k1, k2), yc) in enumerate(zip(rows, ys)):
        y = yc - h / 2
        # 阶段名
        fs.box(ax, 0.012, y, 0.122, h, name, fc="white", ec=col, tc=col,
               fs=8.4, lw=1.3, weight="bold", rounded=0.018)
        # 内容卡
        fs.box(ax, 0.170, y, 0.548, h, body, fc="#FBFBFC", ec="#D8DCE2",
               fs=7.3, lw=0.9, rounded=0.012)
        # 右侧读数
        ax.plot([0.742, 0.742], [y + 0.014, y + h - 0.014], color="#E1E4E9", lw=0.8, zorder=1)
        fs.label(ax, 0.756, yc + 0.026, k1, fs=8.0, color=col, ha="left", weight="bold")
        fs.label(ax, 0.756, yc - 0.026, k2, fs=7.2, color=fs.MUTED, ha="left")

    # 左侧竖向流程箭头
    for i in range(3):
        y0 = ys[i] - h / 2 - 0.004
        y1 = ys[i + 1] + h / 2 + 0.004
        fs.arrow(ax, (0.073, y0), (0.073, y1), color=fs.FAINT, lw=1.0, ms=7)

    return fs.save(fig, "theory", "ch11", "evidence-chain")


def fig_limits():
    """图 8-2：三条硬限制的示意（总体最优／有损观测／边缘不匹配）。"""
    rng = np.random.default_rng(8)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))

    # ---------- (a) 总体最优与有限样本 ----------
    ax = axes[0]
    x = np.linspace(0.02, 1.0, 200)
    lstar = 0.20
    L = lstar + 0.72 * np.exp(-3.0 * x)
    ax.plot(x, L, color=fs.BLUE, lw=1.6, zorder=3)
    ax.axhline(lstar, color=fs.ACCENT, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(0, 1.12); ax.set_ylim(0, 1.0)
    ax.set_xticks([]); ax.set_yticks([])
    fs.clean(ax, spines=("left", "bottom"))

    xm = 0.42
    Lm = lstar + 0.72 * np.exp(-3.0 * xm)
    ax.scatter([xm], [Lm], s=34, color=fs.INK, zorder=5, edgecolor="white", linewidth=1.0)
    ax.annotate("训练停在这里", xy=(xm, Lm), xytext=(0.30, 0.70),
                fontsize=7.6, color=fs.INK,
                arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.8,
                                connectionstyle="arc3,rad=0.18"))
    ax.annotate("总体最优", xy=(0.90, lstar), xytext=(0.52, 0.085),
                fontsize=7.8, color=fs.ACCENT,
                arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8,
                                connectionstyle="arc3,rad=-0.12"))
    fs.label(ax, 0.52, -0.10, "样本数", fs=7.6, color=fs.MUTED)
    fs.label(ax, -0.06, 0.50, "损失", fs=7.6, color=fs.MUTED, rotation=90)
    ax.set_title("(a) 定理说的是那条虚线", fontsize=8.8, color=fs.INK, pad=6)

    # ---------- (b) 单射与有损 ----------
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b) 有损渲染", fontsize=8.8, color=fs.INK, pad=6)

    def state(xx, yy, txt):
        ax.add_patch(Circle((xx, yy), 0.052, facecolor="white", edgecolor=fs.INK,
                            lw=1.1, zorder=4))
        fs.label(ax, xx, yy, txt, fs=9.0)

    def obs(xx, yy, txt):
        ax.add_patch(Rectangle((xx - 0.046, yy - 0.046), 0.092, 0.092, facecolor="white",
                               edgecolor=fs.BLUE, lw=1.1, zorder=4))
        fs.label(ax, xx, yy, txt, fs=8.6, color=fs.BLUE)

    # 左：单射
    state(0.10, 0.735, "$z_1$"); state(0.10, 0.345, "$z_2$")
    obs(0.315, 0.735, "$x_1$"); obs(0.315, 0.345, "$x_2$")
    fs.arrow(ax, (0.155, 0.735), (0.264, 0.735), color=fs.FAINT, lw=1.0, ms=7)
    fs.arrow(ax, (0.155, 0.345), (0.264, 0.345), color=fs.FAINT, lw=1.0, ms=7)
    fs.label(ax, 0.213, 0.545, "单射", fs=7.6, color=fs.MUTED)

    # 右：有损
    state(0.62, 0.735, "$z_1$"); state(0.62, 0.345, "$z_2$")
    obs(0.905, 0.540, "$x$")
    fs.arrow(ax, (0.675, 0.728), (0.858, 0.578), color=fs.ACCENT, lw=1.0, ms=7)
    fs.arrow(ax, (0.675, 0.352), (0.858, 0.502), color=fs.ACCENT, lw=1.0, ms=7)
    fs.label(ax, 0.762, 0.545, "并成一个", fs=7.6, color=fs.ACCENT)
    fs.label(ax, 0.50, 0.080, "两个状态在同一个模态里长得一样", fs=7.6, color=fs.MUTED)

    # ---------- (c) 边缘不相等 ----------
    ax = axes[2]
    p1 = np.array([0.26, 0.20, 0.16, 0.14, 0.12, 0.12])
    p2 = np.array([0.22, 0.24, 0.10, 0.16, 0.19, 0.09])
    xs = np.arange(6)
    ax.bar(xs, p1, width=0.62, color=fs.BLUE, alpha=0.85,
           bottom=0.56, zorder=3, linewidth=0)
    ax.bar(xs, p2, width=0.62, color=fs.ACCENT, alpha=0.85,
           bottom=0.44 - p2, zorder=3, linewidth=0)
    ax.axhline(0.56, color=fs.MUTED, lw=0.7)
    ax.axhline(0.44, color=fs.MUTED, lw=0.7)
    ax.set_xlim(-1.45, 5.6); ax.set_ylim(0, 1.0)
    ax.set_xticks([]); ax.set_yticks([])
    fs.clean(ax, spines=())
    fs.label(ax, -1.34, 0.695, "图像", fs=7.8, color=fs.BLUE, ha="left")
    fs.label(ax, -1.34, 0.285, "文本", fs=7.8, color=fs.ACCENT, ha="left")
    ax.annotate("同一条边质量不同", xy=(2.0, 0.665), xytext=(2.05, 0.925),
                fontsize=7.6, color=fs.INK,
                arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.8))
    fs.label(ax, 2.0, 0.075, "边缘同分布是定理一的第三条前提", fs=7.6, color=fs.MUTED)
    ax.set_title("(c) 两个模态的边缘", fontsize=8.8, color=fs.INK, pad=6)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.05, wspace=0.16)
    return fs.save(fig, "theory", "ch11", "three-limits")


def fig_limits_map():
    """图 11-3：三条限制与四条定理前提的对应矩阵。"""
    fig, ax = fs.canvas(7.0, 3.66)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    FAIL_FC, FAIL_EC = "#FBEAE3", fs.ACCENT
    SHIFT_FC, SHIFT_EC = "#FDF4E4", fs.GOLD
    SAFE_FC, SAFE_EC = "#EDF5F1", fs.GREEN

    cols = [("定理一", "对齐"), ("定理二", "核失配"),
            ("定理三", "干扰的代价"), ("定理四", "视图与数据")]
    rows = [
        ("有损渲染", "前提　$g_m$ 单射", [
            ("单射失效", "换元不再成立", FAIL_FC, FAIL_EC),
            ("前提不破", "失配进分子", SHIFT_FC, SHIFT_EC),
            ("$\\rho$ 掉下来", "下界为正", SHIFT_FC, SHIFT_EC),
            ("$\\tau_m$ 变大", "核不再相同", SHIFT_FC, SHIFT_EC)]),
        ("有限样本", "前提　总体最优", [
            ("唯一性", "只是总体的", FAIL_FC, FAIL_EC),
            ("间隙须", "经验重估", SHIFT_FC, SHIFT_EC),
            ("$\\rho$ 只能估", "且带偏差", SHIFT_FC, SHIFT_EC),
            ("$\\delta_m$ 与 $\\tau_m$", "都不可知", SHIFT_FC, SHIFT_EC)]),
        ("边缘不匹配", "前提　边缘同为 $p$", [
            ("共享 $\\pi$", "不成立", FAIL_FC, FAIL_EC),
            ("公共子空间", "不存在", FAIL_FC, FAIL_EC),
            ("不涉及", "边缘", SAFE_FC, SAFE_EC),
            ("$\\pi_W$ 边缘", "不再是 $p$", FAIL_FC, FAIL_EC)]),
    ]

    x0, x1 = 0.244, 0.988
    gap = 0.013
    w = (x1 - x0 - gap * 3) / 4
    h = 0.212
    ytop = 0.816

    # 列表头
    for j, (t1, t2) in enumerate(cols):
        cx = x0 + j * (w + gap) + w / 2
        fs.label(ax, cx, 0.930, t1, fs=8.8, color=fs.INK, weight="bold")
        fs.label(ax, cx, 0.872, t2, fs=8.0, color=fs.MUTED)

    # 行标签 + 单元格
    for i, (name, prem, cells) in enumerate(rows):
        y = ytop - h - i * (h + 0.030)
        fs.label(ax, 0.012, y + h * 0.62, name, fs=9.2, color=fs.INK,
                 ha="left", weight="bold")
        fs.label(ax, 0.012, y + h * 0.24, prem, fs=8.0, color=fs.MUTED, ha="left")
        for j, (l1, l2, fc, ec) in enumerate(cells):
            x = x0 + j * (w + gap)
            fs.box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=1.1, rounded=0.014)
            fs.label(ax, x + w / 2, y + h * 0.66, l1, fs=8.0, color=fs.INK)
            fs.label(ax, x + w / 2, y + h * 0.28, l2, fs=8.0, color=fs.MUTED)

    # 图例
    ly = 0.055
    items = [("前提失效，结论不成立", FAIL_FC, FAIL_EC),
             ("前提仍成立，界变松", SHIFT_FC, SHIFT_EC),
             ("不涉及这条限制", SAFE_FC, SAFE_EC)]
    for k, (t, fc, ec) in enumerate(items):
        x = 0.012 + k * 0.292
        ax.add_patch(Rectangle((x, ly - 0.020), 0.040, 0.040, facecolor=fc,
                               edgecolor=ec, lw=1.0, zorder=4))
        fs.label(ax, x + 0.052, ly, t, fs=8.0, color=fs.MUTED, ha="left")

    return fs.save(fig, "theory", "ch11", "limits-map")


def witness_two_states():
    """11.9 用到的两状态例子：边缘不同、条件核相同时两个谱嵌入差多远。"""
    q = 0.05
    best = None
    for p1, p2 in (((0.9, 0.1), (0.9, 0.1)), ((0.9, 0.1), (0.1, 0.9))):
        phis = []
        for p in (p1, p2):
            P = np.array([[p[0] - q, q], [q, p[1] - q]])
            d = np.sqrt(np.array(p))
            K = P / np.outer(d, d)
            mu, V = np.linalg.eigh(K)
            order = np.argsort(mu)[::-1]
            mu, V = mu[order], V[:, order]
            phis.append(V[:, 1] / d)
        p1v = np.array(p1)
        rms = min(np.sqrt(np.sum(p1v * (phis[0] - s * phis[1]) ** 2)) for s in (1, -1))
        print(f"边缘 {p1} 对 {p2}: phi1={np.round(phis[0], 4)}, phi2={np.round(phis[1], 4)}, "
              f"最优旋转后均方根距离 {rms:.4f}")
        if p1 == p2:
            best = rms
    return best


if __name__ == "__main__":
    fig_chain()
    fig_limits()
    fig_limits_map()

    # 正文引用的两个参数量，由本脚本算出
    for da, db in ((768, 384), (5120, 384)):
        print(f"{da}x{db}: {da * db - db * (db + 1) // 2:,} 个参数")

    witness_two_states()
