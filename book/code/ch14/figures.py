# -*- coding: utf-8 -*-
"""第 14 章插图：两个模态差一个旋转、谱嵌入的真实计算、定理 2 的数值验证。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import figstyle as fs


# ---------------------------------------------------------------- 数值工具
def sinkhorn(M, p, itmax=60000, tol=1e-13):
    """对称缩放：把对称非负矩阵 M 变成行和为 p 的对称矩阵。"""
    a = np.ones(len(p))
    Mo = M.copy()
    np.fill_diagonal(Mo, 0.0)
    Mo = Mo / Mo.sum()
    for _ in range(itmax):
        R = Mo * a[:, None] * a[None, :]
        row = R.sum(1)
        a = a * np.sqrt(p / row)
        if np.abs(row - p).max() < tol:
            break
    return Mo * a[:, None] * a[None, :]


def normalized_adjacency(Pi, p):
    return Pi / np.sqrt(np.outer(p, p))


def spectrum(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def procrustes_error(A, B):
    """min over Q in O(k) of ||A - B Q^T||_F."""
    U, s, Vt = np.linalg.svd(A.T @ B)
    Q = U @ Vt
    return np.linalg.norm(A - B @ Q.T), Q


# ---------------------------------------------------------------- 图 14-1
def fig_two_modalities():
    fig = plt.figure(figsize=(7.0, 3.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- 左列：两个模态各自的观测
    fs.label(ax, 0.115, 0.955, "两个模态各自的观测", fs=8.6, color=fs.MUTED)
    for i, x in enumerate((0.035, 0.093, 0.151)):
        # 照片：白底 + 一个红圆 + 一个蓝三角
        ax.add_patch(Rectangle((x, 0.655), 0.05, 0.075, facecolor="#FFFFFF",
                               edgecolor=fs.FAINT, lw=0.8, zorder=3))
        ax.add_patch(Circle((x + 0.018, 0.700), 0.011, facecolor=fs.ACCENT,
                            edgecolor="none", alpha=0.85, zorder=4))
        ax.add_patch(Polygon([[x + 0.031, 0.675], [x + 0.043, 0.675], [x + 0.037, 0.690]],
                             closed=True, facecolor=fs.BLUE, edgecolor="none",
                             alpha=0.85, zorder=4))
        # 文字：几条灰线
        ax.add_patch(Rectangle((x, 0.195), 0.05, 0.075, facecolor="#FFFFFF",
                               edgecolor=fs.FAINT, lw=0.8, zorder=3))
        for j, w in enumerate((0.036, 0.030, 0.038)):
            ax.plot([x + 0.007, x + 0.007 + w], [0.248 - j * 0.017] * 2,
                    color=fs.FAINT, lw=1.6, solid_capstyle="round", zorder=4)
    fs.label(ax, 0.115, 0.795, "模态 A　照片", fs=9.0)
    fs.label(ax, 0.115, 0.335, "模态 B　文字", fs=9.0)
    fs.label(ax, 0.115, 0.605, "$x = g_A(z)$", fs=8.2, color=fs.MUTED)
    fs.label(ax, 0.115, 0.145, "$x = g_B(z)$", fs=8.2, color=fs.MUTED)

    # ---- 两个编码器
    fs.box(ax, 0.228, 0.622, 0.112, 0.145, "", fc="#F4F7FA", ec=fs.INK, lw=1.1)
    fs.label(ax, 0.284, 0.723, "编码器 $f_A$", fs=9.2, weight="bold")
    fs.label(ax, 0.284, 0.675, "只看照片", fs=7.8, color=fs.MUTED)
    fs.box(ax, 0.228, 0.162, 0.112, 0.145, "", fc="#F4F7FA", ec=fs.INK, lw=1.1)
    fs.label(ax, 0.284, 0.263, "编码器 $f_B$", fs=9.2, weight="bold")
    fs.label(ax, 0.284, 0.215, "只看文字", fs=7.8, color=fs.MUTED)
    fs.arrow(ax, (0.203, 0.692), (0.226, 0.692), color=fs.INK, lw=1.1)
    fs.arrow(ax, (0.203, 0.232), (0.226, 0.232), color=fs.INK, lw=1.1)

    # ---- 两个小坐标图：同一组点，两套坐标
    rng = np.random.default_rng(11)
    P = np.array([[-0.9, 0.5], [-0.2, 0.9], [0.5, 0.6], [0.9, 0.0],
                  [0.4, -0.6], [-0.3, -0.9], [-0.9, -0.3], [0.0, 0.15]])
    th = np.deg2rad(32.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Pr = P @ R.T

    for pos, pts, col, name in (([0.395, 0.552, 0.155, 0.315], P, fs.BLUE, "A"),
                                ([0.395, 0.092, 0.155, 0.315], Pr, fs.ACCENT, "B")):
        axp = fig.add_axes(pos)
        axp.set_xlim(-1.45, 1.45)
        axp.set_ylim(-1.45, 1.45)
        axp.set_aspect("equal")
        axp.axis("off")
        # 坐标系
        ang = 0.0 if name == "A" else th
        for v in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            vv = R @ v if name == "B" else v
            axp.annotate("", xy=(1.32 * vv[0], 1.32 * vv[1]), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=fs.FAINT, lw=0.9,
                                         mutation_scale=7))
        axp.scatter(pts[:, 0], pts[:, 1], s=26, color=col, zorder=4,
                    edgecolor="white", linewidth=0.6)
        axp.scatter([0], [0], s=4, color=fs.FAINT, zorder=5)
        fs.label(ax, pos[0] + pos[2] / 2, pos[1] - 0.075,
                 "$f_A(g_A(z))$" if name == "A" else "$f_B(g_B(z))$",
                 fs=9.0, color=col)

    fs.arrow(ax, (0.352, 0.692), (0.392, 0.692), color=fs.BLUE, lw=1.1)
    fs.arrow(ax, (0.352, 0.232), (0.392, 0.232), color=fs.ACCENT, lw=1.1)

    # ---- 右列：结论
    ax.plot([0.585, 0.585], [0.10, 0.86], color=fs.GRID, lw=1.0)
    fs.label(ax, 0.618, 0.775, "两个模态从不共享数据，", fs=9.2, ha="left")
    fs.label(ax, 0.618, 0.705, "却落在同一个嵌入上", fs=9.2, ha="left")
    fs.label(ax, 0.618, 0.575, "$f_B(g_B(z)) = Q\\,f_A(g_A(z))$",
             fs=10.5, ha="left", color=fs.INK)
    fs.label(ax, 0.618, 0.470, "$Q$ 是 $k\\times k$ 正交矩阵", fs=8.4, ha="left",
             color=fs.MUTED)
    fs.label(ax, 0.618, 0.408, "同一组点，换了一套坐标", fs=8.4, ha="left", color=fs.MUTED)
    # 角度示意
    ox, oy, ln = 0.640, 0.190, 0.095
    ax.add_patch(plt.Polygon(
        [[ox, oy]] + [[ox + 0.058 * np.cos(a), oy + 0.058 * np.sin(a)]
                      for a in np.linspace(0, th, 24)],
        closed=True, facecolor=fs.ACCENT, alpha=0.16, edgecolor="none", zorder=2))
    ax.plot([ox, ox + ln], [oy, oy], color=fs.INK, lw=1.0, zorder=3)
    ax.plot([ox, ox + ln * np.cos(th)], [oy, oy + ln * np.sin(th)],
            color=fs.INK, lw=1.0, zorder=3)
    ax.plot(ox + 0.058 * np.cos(np.linspace(0, th, 40)),
            oy + 0.058 * np.sin(np.linspace(0, th, 40)),
            color=fs.ACCENT, lw=1.0, zorder=4)
    fs.label(ax, ox + 0.081 * np.cos(th / 2), oy + 0.081 * np.sin(th / 2),
             "$\\theta$", fs=8.2, color=fs.ACCENT)
    fs.label(ax, 0.768, 0.191, "只差一个矩阵", fs=8.4, ha="left",
             color=fs.MUTED)
    return fs.save(fig, "ch14", "two-modalities")


# ---------------------------------------------------------------- 图 14-2
def build_toy_world():
    """12 个状态排成 4x3 的格子，正样本偏好相邻状态。"""
    nx, ny = 4, 3
    n = nx * ny
    xy = np.array([(i, j) for j in range(ny) for i in range(nx)], float)
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    M = np.exp(-d2 / (2 * 0.9 ** 2))
    rng = np.random.default_rng(3)
    p = np.full(n, 1.0 / n) + rng.uniform(-0.03, 0.03, n) / n
    p = p / p.sum()
    Pi = sinkhorn(M, p)
    return xy, p, Pi


def fig_spectral_embedding():
    xy, p, Pi = build_toy_world()
    n = len(p)
    K = normalized_adjacency(Pi, p)
    w, V = spectrum(K)
    k = 2
    Phi = V[:, 1:k + 1] / np.sqrt(p)[:, None]
    print("  图 14-2  特征值", np.round(w[:5], 4),
          " 间隙 mu2-mu3 =", round(w[k] - w[k + 1], 4))

    fig = plt.figure(figsize=(7.0, 2.95))
    axa = fig.add_axes([0.015, 0.09, 0.28, 0.80])
    axb = fig.add_axes([0.355, 0.09, 0.27, 0.80])
    axc = fig.add_axes([0.715, 0.20, 0.265, 0.62])

    # (a) 图本身
    wmax = Pi.max()
    for i in range(n):
        for j in range(i + 1, n):
            if Pi[i, j] > 0.22 * wmax:
                axa.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                         color=fs.BLUE, lw=0.35 + 5.2 * Pi[i, j] / wmax,
                         alpha=0.12 + 0.48 * Pi[i, j] / wmax, zorder=1,
                         solid_capstyle="round")
    axa.scatter(xy[:, 0], xy[:, 1], s=90, color="white", edgecolor=fs.BLUE,
                linewidth=1.1, zorder=3)
    for i in range(n):
        fs.label(axa, xy[i, 0], xy[i, 1], str(i + 1), fs=7.2, color=fs.BLUE, z=4)
    axa.set_xlim(-0.75, 3.75)
    axa.set_ylim(-0.75, 2.75)
    axa.set_aspect("equal")
    axa.set_xticks([])
    axa.set_yticks([])
    fs.clean(axa, spines=())
    fs.label(axa, 1.5, -0.55, "正样本画出的图", fs=8.8)
    fs.label(axa, 1.5, 3.05, "(a)", fs=9, color=fs.MUTED)

    # (b) 谱嵌入
    axb.scatter(Phi[:, 0], Phi[:, 1], s=42, color=fs.ACCENT, zorder=3,
                edgecolor="white", linewidth=0.6)
    for i in range(n):
        axb.annotate(str(i + 1), (Phi[i, 0], Phi[i, 1]), textcoords="offset points",
                     xytext=(5.5, 4.0), fontsize=7.0, color=fs.MUTED, zorder=4)
    axb.set_xlim(-1.75, 1.75)
    axb.set_ylim(-1.95, 1.95)
    axb.set_aspect("equal")
    axb.set_xticks([])
    axb.set_yticks([])
    fs.clean(axb, spines=())
    fs.label(axb, 0.0, -2.42, "前两个特征向量", fs=8.8)
    fs.label(axb, 0.0, 2.24, "(b)", fs=9, color=fs.MUTED)
    fs.label(axb, -1.62, 1.66, "$\\Phi(z)$", fs=10.5, color=fs.ACCENT, ha="left")

    # (c) 特征值
    idx = np.arange(0, 6)
    vals = w[:6]
    axc.axhline(0, color=fs.FAINT, lw=0.8, zorder=1)
    for ii, v in zip(idx, vals):
        col = fs.BLUE if ii == 0 else (fs.ACCENT if ii in (1, 2) else fs.FAINT)
        axc.plot([ii, ii], [0, v], color=col, lw=2.0, zorder=2,
                 solid_capstyle="round")
        axc.scatter([ii], [v], s=26, color=col, zorder=3, edgecolor="white",
                    linewidth=0.6)
    axc.set_xlim(-0.7, 5.6)
    axc.set_ylim(-0.16, 1.30)
    axc.set_xticks(idx)
    axc.set_xticklabels(["$\\mu_0$", "$\\mu_1$", "$\\mu_2$", "$\\mu_3$",
                         "$\\mu_4$", "$\\mu_5$"], fontsize=8.6)
    axc.set_yticks([0, 0.5, 1.0])
    axc.tick_params(axis="y", labelsize=7.5)
    fs.clean(axc)
    # 取用的两个
    axc.plot([1, 1, 2, 2], [0.735, 0.775, 0.775, 0.735], color=fs.ACCENT, lw=0.9)
    fs.label(axc, 1.5, 0.835, "取这两个当坐标", fs=7.2, color=fs.ACCENT)
    # 谱间隙
    axc.annotate("", xy=(2.55, 0.411), xytext=(2.55, 0.118),
                 arrowprops=dict(arrowstyle="<->", color=fs.GREEN, lw=0.9,
                                 shrinkA=0, shrinkB=0))
    fs.label(axc, 2.70, 0.255, "$\\mu_2-\\mu_3$", fs=7.2, color=fs.GREEN, ha="left")
    fs.label(axc, -0.62, 1.13, "常数向量，用不到", fs=6.8, color=fs.MUTED, ha="left")
    fs.label(axc, 3.1, 0.99, "(c)", fs=9, color=fs.MUTED)
    return fs.save(fig, "ch14", "spectral-embedding")


# ---------------------------------------------------------------- 图 14-3
def random_kernel(n, p, rng, clustered):
    if clustered:
        c = rng.integers(0, 2, n)
        M = np.where(c[:, None] == c[None, :],
                     rng.uniform(0.6, 1.0, (n, n)),
                     rng.uniform(0.02, 0.06, (n, n)))
    else:
        M = rng.uniform(0.0, 1.0, (n, n))
    M = (M + M.T) / 2
    return sinkhorn(M, p)


def theorem2_trials(count=80, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    tries = 0
    while len(rows) < count and tries < 20000:
        tries += 1
        n = int(rng.integers(8, 40))
        k = min(int(rng.integers(1, 6)), n - 3)
        p = rng.uniform(0.3, 1.0, n)
        p = p / p.sum()
        clustered = bool(rng.random() < 0.5)
        P1 = random_kernel(n, p, rng, clustered)
        Pb = random_kernel(n, p, rng, False)
        s = np.sqrt(np.outer(p, p))
        K1 = P1 / s
        Kb = Pb / s
        if abs(np.linalg.eigvalsh(K1)[-1] - 1.0) > 1e-8:
            continue
        mix = 10 ** rng.uniform(-4, 0)
        K2 = (1 - mix) * K1 + mix * Kb
        w1, V1 = spectrum(K1)
        w2, V2 = spectrum(K2)
        delta = w2[k] - w2[k + 1]
        if delta <= 1e-6:
            continue
        A = V1[:, 1:k + 1]
        B = V2[:, 1:k + 1]
        err, _ = procrustes_error(A, B)
        diff = K1 - K2
        bound = (2 * np.sqrt(2) / delta) * min(np.sqrt(k) * np.linalg.norm(diff, 2),
                                               np.linalg.norm(diff, "fro"))
        if not np.isfinite(bound) or bound < 1e-10:
            continue
        rows.append((err, bound, clustered, k, n))
    return rows


def fig_theorem2():
    rows = theorem2_trials(80)
    err = np.array([r[0] for r in rows])
    bnd = np.array([r[1] for r in rows])
    clus = np.array([r[2] for r in rows])
    ratio = err / bnd
    print("  图 14-3  点数", len(err), " 误差/上界 最大", round(ratio.max(), 4),
          " 中位", round(np.median(ratio), 4),
          " 误差范围", (round(err.min(), 6), round(err.max(), 3)),
          " 上界范围", (round(bnd.min(), 5), round(bnd.max(), 1)))

    fig, ax = plt.subplots(figsize=(6.5, 3.75))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.955, bottom=0.155)
    lo, hi = 5e-5, 4e3
    ax.plot([lo, hi], [lo, hi], color=fs.INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.scatter(bnd[clus], err[clus], s=26, facecolor="white", edgecolor=fs.GREEN,
               linewidth=0.9, zorder=3, label="聚簇的世界（谱间隙小）")
    ax.scatter(bnd[~clus], err[~clus], s=24, color=fs.BLUE, zorder=3,
               edgecolor="white", linewidth=0.5, label="随机的世界")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(1e-5, 5.0)
    ax.set_xlabel("定理 2 上界，核失配除以谱间隙（对数刻度）", fontsize=9.2, labelpad=5)
    ax.set_ylabel("最优旋转后的误差"
                  "（对数刻度）", fontsize=9.2, labelpad=4)
    fs.clean(ax, grid=True)
    ax.legend(loc="lower right", handletextpad=0.5, borderaxespad=0.9)
    ax.text(1.4e-4, 3.0e0, "每个点是一个随机的有限世界", fontsize=8.4, color=fs.MUTED,
            ha="left", va="top")
    ax.text(1.9e-3, 4.6e-3, "误差 = 上界", fontsize=8.4, color=fs.INK,
            ha="left", va="bottom", rotation=31, rotation_mode="anchor")
    return fs.save(fig, "ch14", "theorem2-check")


if __name__ == "__main__":
    fig_two_modalities()
    fig_spectral_embedding()
    fig_theorem2()
