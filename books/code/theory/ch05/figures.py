# -*- coding: utf-8 -*-
"""第 5 章插图：干扰变量的分解、场景解释比例与误差的对应。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


# ---------------------------------------------------------------- 图 5-1
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
    return fs.save(fig, "theory", "ch05", "nuisance")


# ---------------------------------------------------------------- 图 5-2
def _rho_panel(ax, k, xlim, ylim, yticks, pts, tag):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    fs.clean(ax)

    xs = np.linspace(0.0, 1.0, 500)
    bound = 2 * k * (1 - xs)
    ax.fill_between(xs, 0, bound, color="#F1F1F1", zorder=0, lw=0)
    ax.plot(xs, bound, color=fs.ACCENT, lw=1.4, zorder=2,
            label="定理三下界 $2k(1-\\rho)$")

    for name, rho, err, color, dx, dy, ha in pts:
        b = 2 * k * (1 - rho)
        ax.plot([rho, rho], [b, err], color=color, lw=0.9, ls=(0, (2.2, 1.8)), zorder=3)
        ax.scatter([rho], [err], s=34, color=color, zorder=5,
                   edgecolor="white", linewidth=0.9)
        fs.label(ax, rho + dx, err + dy, f"{name} {err:.1f}",
                 fs=7.9, color=color, ha=ha, va="center")

    ax.legend(loc="lower right", fontsize=7.8, handlelength=1.5, borderpad=0.2,
              labelcolor=fs.ACCENT)
    ax.set_xlabel("场景解释比例 $\\rho$", labelpad=3)
    ax.set_ylabel("绕最优旋转的误差", labelpad=3)
    ax.set_title(tag, loc="left", fontsize=9.6, color=fs.INK, pad=6)
    return ax


def fig_rho_error():
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.45))
    fig.subplots_adjust(wspace=0.34)

    left = [("好视图", .804, 33.5, fs.BLUE, 0.0, 3.9, "center"),
            ("坏视图", .655, 39.4, fs.GOLD, 0.013, 0.0, "left"),
            ("下一词", .608, 40.3, fs.PURPLE, 0.0, 4.0, "center")]
    _rho_panel(axes[0], 32, (0.545, 0.885), (0, 51), [0, 10, 20, 30, 40],
               left, "(a) $k=32$")

    right = [("好视图", .654, 157.0, fs.BLUE, 0.0, 5.0, "center"),
             ("坏视图", .507, 171.5, fs.GOLD, 0.012, 0.0, "left"),
             ("下一词", .440, 176.6, fs.PURPLE, 0.0, 11.0, "center")]
    _rho_panel(axes[1], 128, (0.385, 0.715), (60, 202), [60, 100, 140, 180],
               right, "(b) $k=128$")

    fs.label(axes[0], 0.588, 4.0, "下界以下不可达", fs=7.5, color=fs.MUTED, ha="left")
    return fs.save(fig, "theory", "ch05", "rho-error")


if __name__ == "__main__":
    fig_nuisance()
    fig_rho_error()
