# -*- coding: utf-8 -*-
"""第 2 章插图：三种塌缩状态、奇异值谱对比、塌缩时的三条监控曲线。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    for k, t0 in enumerate((-1.55, 0.0, 1.55)):
        t = t0 + rng.normal(0, 0.38, n)
        pts_dim.append(np.outer(t, u) + np.outer(rng.normal(0, 0.055, n), v))

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
    axes[1].annotate("所有点重合", xy=(0.06, 0.06), xytext=(0.75, 1.55),
                     fontsize=8.2, color=fs.MUTED, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.9,
                                     shrinkA=1, shrinkB=3))

    # (c)
    axes[2].plot([-3.15 * u[0], 3.15 * u[0]], [-3.15 * u[1], 3.15 * u[1]],
                 color=fs.FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for k in range(3):
        axes[2].scatter(pts_dim[k][:, 0], pts_dim[k][:, 1], s=11, color=cols[k],
                        alpha=0.85, linewidths=0, zorder=3)

    return fs.save(fig, "practice", "ch02", "three-states")


def fig_spectrum():
    """三条奇异值谱：健康、维度塌缩、完全塌缩。曲线按典型形状合成，不是论文读数。"""
    d = 2048
    k = np.arange(1, d + 1)
    t = k / d

    healthy = 8.6 - 2.6 * t ** 0.55
    collapse = np.maximum(8.6 - 0.75 * np.sqrt(k), -2.0)
    full = np.full_like(t, -6.0)

    def eff_rank(sigma):
        p = sigma / sigma.sum()
        return float(np.exp(-(p * np.log(p + 1e-300)).sum()))

    er_healthy = eff_rank(10 ** healthy)
    er_collapse = eff_rank(10 ** collapse)
    er_full = eff_rank(10 ** full)
    print(f"  演示谱线有效秩: 健康 {er_healthy:.1f}, 维度塌缩 {er_collapse:.1f}, 完全塌缩 {er_full:.1f}")

    fig = plt.figure(figsize=(6.9, 3.55))
    ax = fig.add_axes([0.088, 0.145, 0.888, 0.815])
    ax.set_xlim(0, d)
    ax.set_ylim(-7.4, 9.4)

    ax.plot(k, healthy, color=fs.BLUE, lw=1.9, zorder=4)
    ax.plot(k, collapse, color=fs.ACCENT, lw=1.9, zorder=5)
    ax.plot(k, full, color=fs.FAINT, lw=1.9, ls=(0, (5, 2.5)), zorder=3)

    ax.axhline(0.0, color=fs.FAINT, lw=0.8, ls=(0, (1.5, 2.5)), zorder=1)
    ax.text(2050, -0.75, "对数零点（奇异值为 1）", fontsize=7.6, color=fs.FAINT, ha="right", va="top")

    fs.clean(ax, grid=True)
    ax.set_xlabel("奇异值序号（按从大到小排列）", fontsize=9.4, color=fs.INK, labelpad=5)
    ax.set_ylabel("奇异值的对数", fontsize=9.4, color=fs.INK, labelpad=6)
    ax.set_xticks([0, 512, 1024, 1536, 2048])
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6, 8])
    ax.tick_params(labelsize=8.4, length=3, colors=fs.MUTED)

    handles = [Line2D([], [], color=fs.BLUE, lw=1.9, label="有投影头"),
               Line2D([], [], color=fs.ACCENT, lw=1.9, label="无投影头"),
               Line2D([], [], color=fs.FAINT, lw=1.9, ls=(0, (5, 2.5)), label="等幅")]
    leg = ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.665, 0.545),
                    handlelength=2.0, handletextpad=0.7, labelspacing=0.55,
                    borderpad=0.2, fontsize=9.0)
    for txt in leg.get_texts():
        txt.set_color(fs.INK)

    fs.label(ax, 300, -3.75, "两百维之后压到 1 以下，其余维度几乎不再变化", fs=8.6,
             color=fs.ACCENT, ha="left")
    ax.annotate("", xy=(214, -1.92), xytext=(240, -3.35),
                arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                shrinkA=3, shrinkB=2))

    fs.label(ax, 2015, -7.0, "演示数据，非论文实测读数", fs=7.8, color=fs.FAINT, ha="right")

    return fs.save(fig, "practice", "ch02", "spectrum")


def fig_monitors():
    """停梯度有无时的三条曲线：训练损失、通道标准差、kNN 准确率。按论文曲线的形状重绘。"""
    d = 2048
    target = 1.0 / np.sqrt(d)
    ep = np.linspace(0, 100, 240)

    loss_ok = -1.0 + 0.95 * np.exp(-ep / 26.0) + 0.06 * np.exp(-ep / 3.0)
    loss_bad = -1.0 + 0.60 * np.exp(-ep / 1.1) + 0.02

    std_ok = target * (0.72 + 0.30 * (1 - np.exp(-ep / 8.0)))
    std_bad = target * 0.42 * np.exp(-ep / 2.2) + 0.0004

    knn_ok = 56.0 * (1 - np.exp(-ep / 34.0))
    knn_bad = np.full_like(ep, 0.35) + 0.25 * np.exp(-ep / 40.0)

    print(f"  1/sqrt(d) = {target:.4f}（d={d}）")

    fig = plt.figure(figsize=(7.0, 3.25))
    axes = [fig.add_axes([0.075, 0.255, 0.252, 0.705]),
            fig.add_axes([0.400, 0.255, 0.252, 0.705]),
            fig.add_axes([0.725, 0.255, 0.252, 0.705])]

    # (a) 训练损失
    ax = axes[0]
    ax.plot(ep, loss_ok, color=fs.BLUE, lw=1.8, zorder=4)
    ax.plot(ep, loss_bad, color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), zorder=4)
    ax.set_ylim(-1.12, 0.10)
    ax.set_yticks([-1.0, -0.5, 0.0])
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("训练损失", fontsize=9.0, labelpad=5)
    fs.label(ax, 4, -0.44, "两条曲线都收敛了", fs=8.4, color=fs.MUTED, ha="left")

    # (b) 通道标准差
    ax = axes[1]
    ax.plot(ep, std_ok, color=fs.BLUE, lw=1.8, zorder=4)
    ax.plot(ep, std_bad, color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), zorder=4)
    ax.axhline(target, color=fs.FAINT, lw=0.9, ls=(0, (1.5, 2.5)), zorder=1)
    ax.set_ylim(0, 0.030)
    ax.set_yticks([0, 0.01, 0.02, 0.03])
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("各通道标准差", fontsize=9.0, labelpad=5)
    fs.label(ax, 4, 0.0268, r"$1/\sqrt{d}$", fs=9.2, color=fs.MUTED, ha="left")
    fs.label(ax, 20, 0.0072, "一路滑到零", fs=8.4, color=fs.ACCENT, ha="left")

    # (c) kNN
    ax = axes[2]
    ax.plot(ep, knn_ok, color=fs.BLUE, lw=1.8, zorder=4)
    ax.plot(ep, knn_bad, color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), zorder=4)
    ax.set_ylim(0, 66)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("kNN 准确率（%）", fontsize=9.0, labelpad=5)
    fs.label(ax, 4, 12.5, "贴着零线", fs=8.4, color=fs.ACCENT, ha="left")

    tags = ["(a)", "(b)", "(c)"]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(labelsize=8.2, length=2.8, colors=fs.MUTED)
        fs.clean(ax, grid=True)
        ax.set_title(tags[i], fontsize=9.4, color=fs.INK, pad=4, loc="left", weight="bold")

    handles = [Line2D([], [], color=fs.BLUE, lw=1.8, label="有停梯度"),
               Line2D([], [], color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), label="去掉停梯度")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False, fontsize=9.0, handlelength=2.2,
               handletextpad=0.7, columnspacing=2.4)

    return fs.save(fig, "practice", "ch02", "monitors")


if __name__ == "__main__":
    fig_states()
    fig_spectrum()
    fig_monitors()
