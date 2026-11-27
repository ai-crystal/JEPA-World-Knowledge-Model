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


def fig_landscape():
    """损失面示意：只有一个不变项时是一整片平坦谷底，加上信息约束后谷底移到有结构的一侧。"""
    gx, gy = np.meshgrid(np.linspace(-2.60, 2.60, 560), np.linspace(-2.10, 2.10, 420))

    # (a) 只有不变项：中间的谷底近乎水平，损失只在外缘才抬起来
    za = 0.04 + 0.42 * ((np.abs(gx) / 2.35) ** 6 + (np.abs(gy) / 1.95) ** 6)
    # (b) 加上信息约束：谷底移到右侧，塌缩一侧被抬成陡坡
    zb = (0.05 + 0.46 * (((gx - 0.78) / 1.45) ** 2 + (gy / 1.45) ** 2)
          + 1.30 * np.exp(-((gx + 1.78) / 0.34) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.170, top=0.905, wspace=0.075)

    la = np.linspace(0.04, 0.62, 11)
    axes[0].contourf(gx, gy, za, levels=la, cmap="Blues_r", alpha=0.90, extend="max", zorder=1)
    axes[0].contour(gx, gy, za, levels=la, colors="#8496AD", linewidths=0.45, zorder=2)

    lb = np.linspace(0.05, 1.95, 13)
    axes[1].contourf(gx, gy, zb, levels=lb, cmap="Blues_r", alpha=0.90, extend="max", zorder=1)
    axes[1].contour(gx, gy, zb, levels=lb, colors="#8496AD", linewidths=0.45, zorder=2)

    for ax in axes:
        ax.set_xlim(-2.60, 2.60)
        ax.set_ylim(-2.10, 2.10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_color(fs.FAINT)
        ax.set_xlabel("参数方向：嵌入越往右越伸展开", fontsize=8.4, color=fs.MUTED, labelpad=4)

    axes[0].set_title("(a) 只有不变项", fontsize=9.6, color=fs.INK, pad=5)
    axes[1].set_title("(b) 加入信息约束", fontsize=9.6, color=fs.INK, pad=5)
    axes[0].set_ylabel("参数的另一个方向", fontsize=8.4, color=fs.MUTED, labelpad=4)

    # (a) 整个谷底一样低，塌缩点只是其中一个位置
    axes[0].text(0.35, 1.28, "整片谷底都接近零，梯度也接近零",
                 fontsize=8.4, color=fs.INK, ha="center", va="center",
                 bbox=dict(fc="white", ec="none", alpha=0.80, pad=2.4), zorder=8)
    axes[0].plot([-1.75], [0.0], marker="o", ms=6.5, mfc="white", mec=fs.ACCENT,
                 mew=1.5, zorder=7)
    axes[0].annotate("常数解", xy=(-1.75, -0.14), xytext=(-2.42, -1.52),
                     fontsize=8.6, color=fs.ACCENT, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                     shrinkA=2, shrinkB=4, mutation_scale=8), zorder=7)

    # (b) 谷底在右侧，塌缩一侧是陡坡
    axes[1].annotate("谷底移到有结构的一侧", xy=(0.78, -0.12), xytext=(0.55, -1.62),
                     fontsize=8.4, color=fs.ACCENT, ha="center", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.1,
                                     shrinkA=2, shrinkB=3, mutation_scale=8), zorder=6)
    axes[1].annotate("塌缩区被抬成陡坡", xy=(-1.80, 0.62), xytext=(-2.42, 1.62),
                     fontsize=8.4, color=fs.INK, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=1.0,
                                     shrinkA=2, shrinkB=3, mutation_scale=8), zorder=6)
    axes[1].plot([-1.75], [0.0], marker="o", ms=6.5, mfc="white", mec=fs.ACCENT,
                 mew=1.5, zorder=7)
    axes[1].annotate("常数解", xy=(-1.75, -0.14), xytext=(-2.42, -1.30),
                     fontsize=8.6, color=fs.FAINT, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.8,
                                     shrinkA=2, shrinkB=4), zorder=8)

    # (b) 谷底在右侧，塌缩一侧是陡坡
    axes[1].annotate("谷底移到有结构的一侧", xy=(0.78, -0.12), xytext=(0.55, -1.62),
                     fontsize=8.4, color=fs.ACCENT, ha="center", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.1,
                                     shrinkA=2, shrinkB=3, mutation_scale=8), zorder=6)
    axes[1].annotate("塌缩区被抬成陡坡", xy=(-1.80, 0.62), xytext=(-2.42, 1.62),
                     fontsize=8.4, color=fs.INK, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=1.0,
                                     shrinkA=2, shrinkB=3, mutation_scale=8), zorder=6)
    axes[1].plot([-1.75], [0.0], marker="o", ms=6.5, mfc="white", mec=fs.ACCENT,
                 mew=1.5, zorder=7)

    return fs.save(fig, "basics", "ch04", "loss-landscape")


def effective_rank(Z, center=True):
    """有效秩：归一化奇异值的熵再取指数。center=False 用于演示不减均值的后果。"""
    if center:
        Z = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def rank_probe():
    """正文引用的几个有效秩参照值，全部在这里算出。"""
    rng = np.random.default_rng(20)
    low = np.zeros((512, 128))
    low[:, :8] = rng.normal(0.0, 1.0, (512, 8))
    print("  有效秩 8 个方向:", round(effective_rank(low), 2))

    d = np.arange(1, 129)
    dec = np.random.default_rng(21).normal(0.0, 1.0, (512, 128)) * 0.7 ** d
    print("  有效秩 按 0.7 衰减:", round(effective_rank(dec), 2))

    rank1 = (np.random.default_rng(22).normal(0.0, 1.0, (512, 1))
             @ np.random.default_rng(23).normal(0.0, 1.0, (1, 128)))
    print("  有效秩 秩为 1:", round(effective_rank(rank1), 2))

    full = np.random.default_rng(24).normal(0.0, 1.0, (512, 128))
    print("  有效秩 满 128 维:", round(effective_rank(full), 2))
    print("  有效秩 不减均值（整体平移 8）:", round(effective_rank(full + 8.0, center=False), 2))


def fig_effective_rank():
    """有效秩随训练步数的三条曲线（示意数据，按各类方法的典型行为构造）。"""
    d = 256
    rng = np.random.default_rng(7)
    steps = np.logspace(2, np.log10(60000), 420)

    def smooth(y, k=9):
        w = np.ones(k) / k
        return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), w, mode="valid")

    bad = d * (0.93 * np.exp(-steps / 3200) + 0.008)
    neg = d * (0.735 + 0.175 * np.exp(-steps / 9000))
    wht = d * (0.800 + 0.075 * np.exp(-steps / 6000) + 0.0000011 * steps)

    bad = smooth(bad + rng.normal(0, 1.1, steps.size), 13)
    neg = smooth(neg + rng.normal(0, 2.4, steps.size), 19)
    wht = smooth(wht + rng.normal(0, 2.0, steps.size), 19)

    fig = plt.figure(figsize=(6.9, 3.25))
    ax = fig.add_axes([0.098, 0.185, 0.870, 0.760])
    ax.axhline(d, color=fs.FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.plot(steps, wht, color=fs.BLUE, lw=1.9, zorder=4)
    ax.plot(steps, neg, color=fs.GREEN, lw=1.9, zorder=4)
    ax.plot(steps, bad, color=fs.ACCENT, lw=1.9, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(1e2, 6.4e4)
    ax.set_ylim(0, 292)
    ax.set_yticks([0, 64, 128, 192, 256])
    ax.set_xticks([1e2, 1e3, 1e4, 6e4])
    ax.set_xticklabels(["100", "1k", "10k", "60k"])
    ax.set_xlabel("训练步数（对数刻度）", fontsize=9.2, labelpad=5)
    ax.set_ylabel("有效秩（相当于几个方向）", fontsize=9.2, labelpad=5)
    fs.clean(ax, grid=True)

    ax.text(1.1e2, d + 8, "维度上限 256", fontsize=8.3, color=fs.FAINT, ha="left", va="bottom")
    ax.text(6.2e4, wht[-1] + 12, "白化", fontsize=9.0, color=fs.BLUE, ha="right", va="bottom")
    ax.text(6.2e4, neg[-1] - 14, "负样本", fontsize=9.0, color=fs.GREEN, ha="right", va="top")
    ax.annotate("无约束", xy=(4.2e4, bad[-1] + 4), xytext=(1.05e4, 30),
                fontsize=9.0, color=fs.ACCENT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.0,
                                shrinkA=3, shrinkB=2, mutation_scale=8))

    rank_probe()
    return fs.save(fig, "basics", "ch04", "effective-rank")


if __name__ == "__main__":
    fig_states()
    fig_infonce()
    fig_landscape()
    fig_effective_rank()
