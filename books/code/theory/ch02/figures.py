# -*- coding: utf-8 -*-
"""第 2 章插图：两个模态落到同一个嵌入（图 2-1）、谱嵌入的真实计算（图 2-2）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
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


# ---------------------------------------------------------------- 图 2-1
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
    return fs.save(fig, "theory", "ch02", "two-modalities")


# ---------------------------------------------------------------- 图 2-2
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
    print("  图 2-2  特征值", np.round(w[:5], 4),
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
    return fs.save(fig, "theory", "ch02", "spectral-embedding")



# ---------------------------------------------------------------- 图 2-3
def fig_spectrum_map():
    """K 的特征值与 L = I - K 的特征值落在直线 lambda = 1 - mu 上。"""
    xy, p, Pi = build_toy_world()
    K = normalized_adjacency(Pi, p)
    w, V = spectrum(K)
    lam = 1.0 - w
    print("  图 2-3  mu", np.round(w[:6], 4), "lambda", np.round(lam[:6], 4))

    fig = plt.figure(figsize=(6.5, 3.55))
    ax = fig.add_axes([0.125, 0.155, 0.845, 0.795])

    ax.axvspan(-1.12, 0.0, facecolor=fs.GRID, alpha=0.55, zorder=0)
    ax.axhline(1.0, color=fs.FAINT, lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax.axvline(0.0, color=fs.FAINT, lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax.plot([-1.0, 1.0], [2.0, 0.0], color=fs.INK, lw=1.1, zorder=2)

    # 12 个点全部画出，只给前四个加标注
    ax.scatter(w, lam, s=20, color=fs.FAINT, zorder=3, edgecolor="none")
    idx = [0, 1, 2, 3]
    cols = [fs.BLUE, fs.ACCENT, fs.ACCENT, fs.ACCENT]
    offs = {0: (9, -6, "left"), 1: (9, 14, "left"),
            2: (-12, -15, "right"), 3: (12, -18, "left")}
    for i, c in zip(idx, cols):
        ax.scatter([w[i]], [lam[i]], s=38, color=c, zorder=4,
                   edgecolor="white", linewidth=0.7)
        dx, dy, ha = offs[i]
        txt = "$\\mu_0$（常数）" if i == 0 else "$\\mu_%d,\\ \\lambda_%d$" % (i, i)
        ax.annotate(txt, (w[i], lam[i]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.8, color=c, ha=ha, zorder=5)

    ax.annotate("其余 $8$ 个", xy=(w[-1], lam[-1]), xytext=(-0.88, 1.52),
                fontsize=7.6, color=fs.MUTED, ha="center",
                arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.7,
                                shrinkA=2, shrinkB=4))

    # 谱间隙
    ax.annotate("", xy=(w[3], 1.0), xytext=(w[2], 1.0),
                arrowprops=dict(arrowstyle="<->", color=fs.GREEN, lw=0.9,
                                shrinkA=0, shrinkB=0))
    fs.label(ax, 0.5 * (w[2] + w[3]) + 0.05, 1.05, "间隙 $0.294$", fs=7.8,
             color=fs.GREEN, va="bottom", ha="left")

    fs.label(ax, 0.70, 1.80, "$\\lambda = 1-\\mu$", fs=9.4, color=fs.INK)
    fs.label(ax, -1.06, 0.34, "$\\mu$ 为负时 $\\lambda$ 超过 $1$", fs=7.8,
             color=fs.MUTED, ha="left")

    ax.set_xlim(-1.12, 1.30)
    ax.set_ylim(-0.16, 2.12)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(labelsize=8.0)
    ax.set_xlabel("$K$ 的特征值 $\\mu$", fontsize=9.2)
    ax.set_ylabel("$L$ 的特征值 $\\lambda$", fontsize=9.2)
    fs.clean(ax)
    return fs.save(fig, "theory", "ch02", "spectrum-map")


# ---------------------------------------------------------------- 图 2-4
def fig_small_world_3d():
    """12 状态小世界的前三个谱坐标：三维里是一张马鞍面，俯视是棋盘。"""
    xy, p, Pi = build_toy_world()
    n = len(p)
    K = normalized_adjacency(Pi, p)
    w, V = spectrum(K)
    Phi = V[:, 1:4] / np.sqrt(p)[:, None]

    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    print("  图 2-4  前三维坐标范围",
          np.round([Phi.min(0), Phi.max(0)], 3).tolist())

    fig = plt.figure(figsize=(6.9, 3.45))
    ax = fig.add_axes([0.0, 0.0, 0.585, 0.98], projection="3d")
    axb = fig.add_axes([0.615, 0.155, 0.365, 0.735])

    # ---- 格子边（正交相邻）与斜边分开画
    for i in range(n):
        for j in range(i + 1, n):
            if d2[i, j] == 1.0:
                ax.plot([Phi[i, 0], Phi[j, 0]], [Phi[i, 1], Phi[j, 1]],
                        [Phi[i, 2], Phi[j, 2]], color=fs.BLUE, lw=0.7,
                        alpha=0.35, zorder=1)
    # ---- φ1-φ2 平面上的投影，作为参照
    z0 = -1.85
    for i in range(n):
        for j in range(i + 1, n):
            if d2[i, j] == 1.0:
                ax.plot([Phi[i, 0], Phi[j, 0]], [Phi[i, 1], Phi[j, 1]], [z0, z0],
                        color=fs.GRID, lw=0.7, zorder=0)
        ax.plot([Phi[i, 0], Phi[i, 0]], [Phi[i, 1], Phi[i, 1]],
                [z0, Phi[i, 2]], color=fs.GRID, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.scatter(Phi[:, 0], Phi[:, 1], np.full(n, z0), s=12, color=fs.FAINT,
               edgecolor="none", depthshade=False, zorder=0)

    sgn = np.sign(Phi[:, 2])
    ax.scatter(Phi[:, 0], Phi[:, 1], Phi[:, 2], s=40,
               color=[fs.ACCENT if s > 0 else fs.GREEN for s in sgn],
               edgecolor="white", linewidth=0.6, depthshade=False, zorder=3)
    ax.set_xlabel("$\\varphi_1$", fontsize=8.4, labelpad=-4)
    ax.set_ylabel("$\\varphi_2$", fontsize=8.4, labelpad=-4)
    ax.set_zlabel("$\\varphi_3$", fontsize=8.4, labelpad=-5)
    ax.tick_params(labelsize=6.5, pad=-2)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_zticks([-1.0, 0.0, 1.0])
    ax.set_zlim(-1.95, 1.72)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(fs.GRID)
    ax.grid(True)
    ax.set_box_aspect((1.25, 1.0, 0.85))
    ax.view_init(elev=19, azim=-63)
    fig.text(0.285, 0.018, "(a) 前三个坐标", fontsize=8.6, ha="center")

    # ---- 俯视：第三维只由前两维决定，符号是棋盘
    for i in range(n):
        for j in range(i + 1, n):
            if d2[i, j] == 1.0:
                axb.plot([Phi[i, 0], Phi[j, 0]], [Phi[i, 1], Phi[j, 1]],
                         color=fs.BLUE, lw=0.7, alpha=0.30, zorder=1)
    for s, c, lab in ((1, fs.ACCENT, "$\\varphi_3>0$"),
                      (-1, fs.GREEN, "$\\varphi_3<0$")):
        m = sgn == s
        axb.scatter(Phi[m, 0], Phi[m, 1], s=50, color=c, edgecolor="white",
                    linewidth=0.7, zorder=3, label=lab)
    for i in range(n):
        axb.annotate(str(i + 1), (Phi[i, 0], Phi[i, 1]), textcoords="offset points",
                     xytext=(6.0, 4.5), fontsize=7.0, color=fs.MUTED, zorder=4)
    axb.set_xlim(-2.05, 2.30)
    axb.set_ylim(-2.30, 2.05)
    axb.set_aspect("equal")
    axb.set_xticks([])
    axb.set_yticks([])
    fs.clean(axb, spines=())
    axb.legend(loc="lower center", fontsize=7.6, handletextpad=0.3, ncol=2,
               bbox_to_anchor=(0.5, -0.16))
    fs.label(axb, 0.05, 1.90, "$\\varphi_2$", fs=9.0, color=fs.MUTED)
    fs.label(axb, 2.10, -0.02, "$\\varphi_1$", fs=9.0, color=fs.MUTED)
    fs.label(axb, 0.10, -2.16, "(b) 按第三维的符号着色", fs=8.0)
    return fs.save(fig, "theory", "ch02", "small-world")


if __name__ == "__main__":
    fig_two_modalities()
    fig_spectral_embedding()
    fig_spectrum_map()
    fig_small_world_3d()
