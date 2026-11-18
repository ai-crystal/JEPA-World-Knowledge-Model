# -*- coding: utf-8 -*-
"""第 4 章插图：四种嵌入形状、随机切片投影、Epps-Pulley 统计量曲线。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import figstyle as fs


# ---------------- 图 4-1 四种二维嵌入 ----------------
def fig_four_states():
    rng = np.random.default_rng(7)
    n = 1400

    # (a) 完全塌缩：所有点挤在原点附近
    a = rng.normal(0.0, 0.012, size=(n, 2))
    # (b) 维度塌缩：全部落在一个方向上，沿该方向方差为 1
    t = rng.normal(0.0, 1.0, size=n)
    ang = np.deg2rad(32.0)
    b = np.stack([t * np.cos(ang), t * np.sin(ang)], axis=1)
    # (c) 白化但非高斯：协方差恰为单位阵，支撑在两条对角线上
    z = rng.normal(0.0, 1.0, size=n)
    flip = rng.choice([-1.0, 1.0], size=n)
    c = np.stack([z, flip * z], axis=1)
    # (d) 各向同性高斯
    d = rng.normal(0.0, 1.0, size=(n, 2))

    sets = [(a, "(a) 完全塌缩", fs.ACCENT), (b, "(b) 维度塌缩", fs.GOLD),
            (c, "(c) 白化后的 X 形", fs.PURPLE), (d, "(d) 各向同性高斯", fs.BLUE)]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.95))
    for ax, (pts, title, col) in zip(axes, sets):
        for r in (1.0, 2.0):
            ax.add_patch(Circle((0, 0), r, fill=False, ec="#E5E7EB", lw=0.7,
                                ls=(0, (3, 3)), zorder=1))
        ax.scatter(pts[:, 0], pts[:, 1], s=4.5, color=col, alpha=0.45,
                   linewidths=0, zorder=3)
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelsize=8, length=2.5, pad=2)
        fs.clean(ax)
        ax.set_title(title, fontsize=9, color=fs.INK, pad=5)
        cov = np.cov(pts.T)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        print(f"  {title}: 特征值 {ev[0]:.3f} / {ev[1]:.3f}")

    # (c)(d) 面板底部各加一句特征值说明会挤，统一交给图注
    fig.subplots_adjust(wspace=0.28, left=0.045, right=0.985, top=0.90, bottom=0.10)
    return fs.save(fig, "practice", "ch04", "four-states")


# ---------------- 图 4-2 随机切片投影 ----------------
def _x_sample(rng, n):
    z = rng.normal(0.0, 1.0, size=n)
    flip = rng.choice([-1.0, 1.0], size=n)
    return np.stack([z, flip * z], axis=1)


def _kde(x, grid, bw=0.11):
    x = np.asarray(x)
    u = (grid[:, None] - x[None, :]) / bw
    k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi) / bw
    return k.mean(axis=1)


def fig_slices():
    rng = np.random.default_rng(11)
    pts = _x_sample(rng, 1500)
    grid = np.linspace(-5.0, 5.0, 600)
    norm_pdf = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)

    fig = plt.figure(figsize=(7.0, 3.95))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.14, 1.0], wspace=0.10,
                          hspace=0.55, left=0.045, right=0.985, top=0.91, bottom=0.12)

    # 左：点云 + 若干投影方向
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(pts[:, 0], pts[:, 1], s=4.0, color=fs.PURPLE, alpha=0.28,
               linewidths=0, zorder=2)
    degs = [0, 20, 45, 70, 90, 115, 135, 160]
    for dg in degs:
        th = np.deg2rad(dg)
        on = dg in (0, 45, 90)
        col = fs.ACCENT if dg == 45 else (fs.MUTED if on else fs.FAINT)
        ax.annotate("", xy=(2.55 * np.cos(th), 2.55 * np.sin(th)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.25 if on else 0.8,
                                    alpha=1.0 if on else 0.55,
                                    mutation_scale=9))
    ax.text(1.72, 1.86, "45°", fontsize=8.2, color=fs.ACCENT)
    ax.text(2.42, -0.42, "0°", fontsize=8.2, color=fs.MUTED)
    ax.text(-0.55, 2.42, "90°", fontsize=8.2, color=fs.MUTED)
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=8, length=2.5, pad=2)
    fs.clean(ax)
    ax.set_title("嵌入点云与切片方向", fontsize=9, pad=5)

    # 右：三个方向的一维分布
    rows = [(0, "沿 0° 投影"), (45, "沿 45° 投影"), (90, "沿 90° 投影")]
    for i, (dg, name) in enumerate(rows):
        axi = fig.add_subplot(gs[i, 1])
        th = np.deg2rad(dg)
        proj = pts[:, 0] * np.cos(th) + pts[:, 1] * np.sin(th)
        dens = _kde(proj, grid, bw=0.22)
        hot = dg == 45
        axi.fill_between(grid, dens, color=fs.ACCENT if hot else fs.BLUE,
                         alpha=0.16, zorder=2)
        axi.plot(grid, dens, color=fs.ACCENT if hot else fs.BLUE, lw=1.3, zorder=3)
        axi.plot(grid, norm_pdf, color=fs.INK, lw=1.0, ls=(0, (4, 2.5)), zorder=4)
        axi.set_xlim(-3.2, 3.2)
        axi.set_ylim(0, max(1.05 * dens.max(), 0.5) * 1.38)
        axi.set_yticks([])
        axi.set_xticks([-2, 0, 2])
        axi.tick_params(labelsize=8, length=2.5, pad=2)
        fs.clean(axi)
        axi.text(-3.05, 0.95 * axi.get_ylim()[1], name, fontsize=8.4,
                 color=fs.INK, ha="left", va="top")
        if i < 2:
            axi.set_xticklabels([])
        print(f"  沿 {dg}° 投影：标准差 {proj.std():.3f}，"
              f"峰度 {((proj - proj.mean()) ** 4).mean() / proj.var() ** 2 - 3:.2f}")

    return fs.save(fig, "practice", "ch04", "slices")


# ---------------- 图 4-3 Epps-Pulley 统计量 ----------------
def epps_pulley(x, n_points=17, lo=-5.0, hi=5.0):
    """与论文 Algorithm 1 一致的实现：高斯窗、梯形积分、区间 [-5, 5]。"""
    t = np.linspace(lo, hi, n_points)
    ecf = np.exp(1j * np.outer(x, t)).mean(axis=0)
    target = np.exp(-0.5 * t ** 2)
    err = np.abs(ecf - target) ** 2 * target
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return x.size * trapz(err, t)


def fig_epps_pulley():
    rng = np.random.default_rng(2024)
    eps = np.linspace(0.0, 1.0, 21)
    n, seeds = 500, 40

    def uniform_unit(k):
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size=k)

    def two_point_unit(k):
        return rng.choice([-1.0, 1.0], size=k)

    def laplace_unit(k):
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=k)

    fams = [("混入均匀分布", uniform_unit, fs.BLUE),
            ("混入两点分布", two_point_unit, fs.GREEN),
            ("混入拉普拉斯分布", laplace_unit, fs.ACCENT)]

    fig, ax = plt.subplots(figsize=(6.6, 3.30))
    base = None
    for name, gen, col in fams:
        mean_curve, sd_curve = [], []
        for e in eps:
            vals = []
            for _ in range(seeds):
                z = rng.normal(0.0, 1.0, size=n)
                u = gen(n)
                x = np.sqrt(1.0 - e) * z + np.sqrt(e) * u
                vals.append(epps_pulley(x))
            mean_curve.append(np.mean(vals))
            sd_curve.append(np.std(vals))
        mean_curve = np.array(mean_curve); sd_curve = np.array(sd_curve)
        ax.fill_between(eps, mean_curve - sd_curve, mean_curve + sd_curve,
                        color=col, alpha=0.13, linewidth=0)
        ax.plot(eps, mean_curve, color=col, lw=1.6, marker="o", ms=2.6,
                label=name, zorder=4)
        if base is None:
            base = mean_curve[0]
        print(f"  {name}: EP(0)={mean_curve[0]:.2f}  EP(1)={mean_curve[-1]:.2f}")

    ax.axhline(base, color=fs.FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.30, base + 2.2, "纯高斯样本的有限样本水平", fontsize=8.2,
            color=fs.MUTED, ha="left", va="bottom")
    ax.set_xlabel("非高斯成分的混合比例 ε")
    ax.set_ylabel("Epps–Pulley 统计量")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, None)
    fs.clean(ax, grid=True)
    ax.legend(loc="upper left", fontsize=8.6, handlelength=1.7, borderaxespad=0.6)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.155)
    return fs.save(fig, "practice", "ch04", "epps-pulley")


if __name__ == "__main__":
    fig_four_states()
    fig_slices()
    fig_epps_pulley()
