# -*- coding: utf-8 -*-
"""第 6 章插图：BYOL/SimSiam 结构、Barlow Twins 互相关矩阵、白化的几何效果。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import figstyle as fs


# ---------- 小工具 ----------
def _scale(ax):
    """返回 (x 方向每数据单位多少英寸, y 方向每数据单位多少英寸)。"""
    fw, fh = ax.figure.get_size_inches()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return fw / (x1 - x0), fh / (y1 - y0)


def stopgrad_arrow(ax, p0, p1, frac=0.62, gap=0.15, half=0.080,
                   color=fs.INK, lw=1.2, ms=9):
    """带断开符号（stop-gradient）的箭头：中间留缺口，缺口上画两道斜杠。"""
    sx, sy = _scale(ax)
    A = np.array([p0[0] * sx, p0[1] * sy])
    B = np.array([p1[0] * sx, p1[1] * sy])
    d = B - A
    u = d / np.linalg.norm(d)
    n = np.array([-u[1], u[0]])
    mid = A + d * frac
    g1, g2 = mid - u * gap / 2, mid + u * gap / 2
    to_d = lambda q: (q[0] / sx, q[1] / sy)
    fs.arrow(ax, p0, to_d(g1), color=color, lw=lw, ms=ms, style="-")
    fs.arrow(ax, to_d(g2), p1, color=color, lw=lw, ms=ms)
    for off in (-0.014, 0.014):
        c = mid + u * off
        a1, a2 = c - n * half / 2, c + n * half / 2
        q1, q2 = to_d(a1), to_d(a2)
        ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color=color, lw=lw * 0.95,
                solid_capstyle="round", zorder=6)
    return to_d(mid)


# ---------- 图 6-1 ----------
def fig_byol():
    fig, ax = fs.canvas(7.0, 3.95)
    y_on, y_tg = 0.760, 0.320
    bh = 0.118

    def row(y, tag, sub, ec):
        fs.box(ax, 0.020, y - bh / 2, 0.098, bh, tag, fc="#FFFFFF", ec=ec, fs=9.0, lw=1.1)
        fs.box(ax, 0.158, y - bh / 2, 0.150, bh, "编码器\n" + sub[0],
               fc="#F6F8FB", ec=ec, fs=8.6, lw=1.2)
        fs.box(ax, 0.348, y - bh / 2, 0.115, bh, "投影头\n" + sub[1],
               fc="#F6F8FB", ec=ec, fs=8.6, lw=1.2)
        fs.arrow(ax, (0.120, y), (0.156, y), color=fs.MUTED, lw=1.1)
        fs.arrow(ax, (0.310, y), (0.346, y), color=fs.MUTED, lw=1.1)

    row(y_on, "视图 1", ("fθ", "gθ"), fs.ACCENT)
    row(y_tg, "视图 2", ("fξ", "gξ"), fs.BLUE)

    # 在线分支的预测器
    fs.box(ax, 0.503, y_on - bh / 2, 0.115, bh, "预测器\nqθ", fc="#FDF1E9",
           ec=fs.ACCENT, fs=8.6, lw=1.2)
    fs.arrow(ax, (0.465, y_on), (0.501, y_on), color=fs.MUTED, lw=1.1)

    # 损失框
    fs.box(ax, 0.740, 0.452, 0.215, 0.190, "", fc="#FFFFFF", ec=fs.INK, lw=1.3)
    fs.label(ax, 0.8475, 0.585, "损失", fs=9.4, weight="bold")
    fs.label(ax, 0.8475, 0.505, "L = || p − z′ ||²", fs=9.6, color=fs.INK)

    # 两条进损失的箭头
    fs.arrow(ax, (0.620, y_on), (0.738, 0.590), color=fs.ACCENT, lw=1.3)
    fs.label(ax, 0.690, 0.716, "p", fs=10.5, color=fs.ACCENT, weight="bold")
    mid = stopgrad_arrow(ax, (0.465, y_tg), (0.738, 0.500), frac=0.60,
                         color=fs.BLUE, lw=1.3)
    fs.label(ax, 0.512, 0.268, "z′", fs=10.5, color=fs.BLUE, weight="bold")
    fs.label(ax, mid[0] + 0.002, mid[1] - 0.068, "停梯度", fs=8.4, color=fs.BLUE)

    # EMA
    fs.arrow(ax, (0.196, y_on - bh / 2 - 0.004), (0.196, y_tg + bh / 2 + 0.004),
             color=fs.GREEN, lw=1.2, ls=(0, (4, 3)), style="-|>", ms=8)
    fs.label(ax, 0.212, 0.552, "指数滑动平均", fs=8.4, color=fs.GREEN, ha="left")
    fs.label(ax, 0.212, 0.492, "ξ 慢速跟随 θ，不接收梯度", fs=8.0, color=fs.GREEN, ha="left")

    # 行注释
    fs.label(ax, 0.158, 0.905, "在线网络：梯度只沿这一条回流", fs=8.6,
             color=fs.ACCENT, ha="left")
    fs.label(ax, 0.158, 0.178, "目标网络：只被复制，不被训练", fs=8.6,
             color=fs.BLUE, ha="left")
    fs.label(ax, 0.500, 0.062,
             "SimSiam 去掉目标网络与滑动平均，两侧共用 θ，只留下预测器和停梯度",
             fs=8.2, color=fs.MUTED)
    return fs.save(fig, "ch06", "byol-simsiam")


# ---------- 图 6-2 ----------
def _xcorr(a, b):
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return a.T @ b / len(a)


def _barlow_matrices(N=4096, D=32, seed=7, noise=0.12):
    """真实算一个 32 维的演示例子。

    两路嵌入共享一个分块的协方差结构（4 组，组内高度相关），各自再叠一点独立噪声。
    左图是直接算出的互相关矩阵；右图先用一半样本估出互协方差矩阵的逆平方根做变换
    （这正是 Barlow Twins 损失的不动点），再在另一半样本上重算，避免自己拟合自己。
    """
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(4), D // 4)
    block = (g[:, None] == g[None, :]).astype(float)
    cov = 0.9 * block + 0.45 * np.ones((D, D)) + 0.30 * np.eye(D)
    sig = rng.normal(size=(N, D)) @ np.linalg.cholesky(cov).T
    za = sig + noise * rng.normal(size=(N, D))
    zb = sig + noise * rng.normal(size=(N, D))

    C0 = _xcorr(za, zb)
    h = N // 2
    cross = za[:h].T @ zb[:h] / h
    cross = (cross + cross.T) / 2
    w, V = np.linalg.eigh(cross)
    W = V @ np.diag(w ** -0.5) @ V.T
    # 每一列拉回单位尺度，这一步对应 Barlow Twins 里的逐维标准化
    W = W / np.sqrt((za[:h] @ W).std(0) * (zb[:h] @ W).std(0))
    C1 = _xcorr(za[h:] @ W, zb[h:] @ W)
    return C0, C1


def fig_barlow():
    C0, C1 = _barlow_matrices()
    D = C0.shape[0]
    off = ~np.eye(D, dtype=bool)
    rms0 = float(np.sqrt((C0[off] ** 2).mean()))
    rms1 = float(np.sqrt((C1[off] ** 2).mean()))
    diag0 = float(np.diag(C0).mean())
    diag1 = float(np.diag(C1).mean())
    print(f"  Barlow 演示：未约束 off-diag RMS={rms0:.3f}, diag 均值={diag0:.3f}")
    print(f"  Barlow 演示：约束后 off-diag RMS={rms1:.3f}, diag 均值={diag1:.3f}")

    fig = plt.figure(figsize=(6.9, 3.15))
    axes = [fig.add_axes([0.058, 0.155, 0.345, 0.740]),
            fig.add_axes([0.478, 0.155, 0.345, 0.740])]
    for ax, C, tag, rms in zip(axes, (C0, C1), ("(a) 未加约束", "(b) 加上冗余消减"),
                               (rms0, rms1)):
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([0, 15, 31]); ax.set_yticks([0, 15, 31])
        ax.tick_params(labelsize=8, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(tag, fontsize=9.4, color=fs.INK, pad=5)
        ax.set_xlabel("嵌入维度 j", fontsize=8.6, labelpad=2)
        ax.set_ylabel("嵌入维度 i", fontsize=8.6, labelpad=2)
        ax.text(0.5, -0.215, f"非对角元均方根 {rms:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.2, color=fs.MUTED)

    cax = fig.add_axes([0.848, 0.155, 0.022, 0.740])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(0.6)
    cb.set_label("互相关 $C_{ij}$", fontsize=8.6, labelpad=6)
    return fs.save(fig, "ch06", "barlow-xcorr")


# ---------- 图 6-3 ----------
def fig_whiten():
    rng = np.random.default_rng(11)
    n = 700
    cov = np.array([[1.85, 1.25], [1.25, 1.05]])
    X = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    w, V = np.linalg.eigh(cov)
    Wm = V @ np.diag(w ** -0.5) @ V.T
    Z = X @ Wm.T
    print(f"  白化演示：原始协方差特征值 {w[0]:.3f}, {w[1]:.3f}")
    print(f"  白化后协方差 = {np.round(np.cov(Z.T), 3).tolist()}")

    fig = plt.figure(figsize=(6.5, 3.05))
    panels = [fig.add_axes([0.075, 0.135, 0.335, 0.755]),
              fig.add_axes([0.565, 0.135, 0.335, 0.755])]
    lim = 3.4
    for ax, P, tag in zip(panels, (X, Z), ("(a) 白化前", "(b) 白化后")):
        ax.scatter(P[:, 0], P[:, 1], s=7, color=fs.BLUE, alpha=0.35,
                   linewidths=0, zorder=2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color=fs.FAINT, lw=0.7, zorder=1)
        ax.axvline(0, color=fs.FAINT, lw=0.7, zorder=1)
        fs.clean(ax, spines=())
        ax.tick_params(labelsize=8, length=0)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
        ax.set_title(tag, fontsize=9.4, pad=5)
        ax.set_xlabel("第 1 维", fontsize=8.6, labelpad=1)
        ax.set_ylabel("第 2 维", fontsize=8.6, labelpad=1)

    # 左：协方差的 1σ 椭圆与两条主轴
    ax = panels[0]
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]),
                         angle=ang, facecolor="none", edgecolor=fs.ACCENT,
                         lw=1.5, zorder=4))
    for k in (0, 1):
        d = V[:, k] * np.sqrt(w[k])
        ax.plot([-d[0], d[0]], [-d[1], d[1]], color=fs.ACCENT, lw=0.9,
                ls=(0, (3, 2)), zorder=5)

    # 右：单位圆
    ax = panels[1]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=fs.GREEN, lw=1.5, zorder=4)
    for d in ((1.0, 0.0), (0.0, 1.0)):
        ax.annotate("", xy=d, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=fs.GREEN, lw=1.0,
                                    shrinkA=0, shrinkB=0, mutation_scale=8), zorder=5)

    notes = [("方差挤在一条斜线上", fs.ACCENT), ("各方向方差为 1，方向互不相关", fs.GREEN)]
    for ax, (txt, col) in zip(panels, notes):
        ax.text(0.13, 0.945, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.0, color=col, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.6))
    return fs.save(fig, "ch06", "whitening")


if __name__ == "__main__":
    fig_byol()
    fig_barlow()
    fig_whiten()
