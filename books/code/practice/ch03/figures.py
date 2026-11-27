# -*- coding: utf-8 -*-
"""第 3 章插图：BYOL/SimSiam 结构、Barlow Twins 互相关矩阵、白化的几何效果。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
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


def stopgrad_arrow(ax, p0, p1, frac=0.62, gap=0.26, half=0.30,
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
    # 缺口处画一道垂直的"挡板"，表示梯度在此被截断
    for off in (-0.030, 0.030):
        c = mid + u * off
        a1, a2 = c - n * half / 2, c + n * half / 2
        q1, q2 = to_d(a1), to_d(a2)
        ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color=color,
                lw=lw * 1.6 if off > 0 else lw * 1.0,
                solid_capstyle="round", zorder=6, alpha=1.0 if off > 0 else 0.45)
    return to_d(mid)


# ---------- 图 3-1 ----------
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
    fs.label(ax, 0.500, 0.268, "z′", fs=10.5, color=fs.BLUE, weight="bold")
    fs.label(ax, mid[0] + 0.004, mid[1] - 0.072, "停梯度", fs=8.4, color=fs.BLUE)

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
    return fs.save(fig, "practice", "ch03", "byol-simsiam")


# ---------- 图 3-2 ----------
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

    fig = plt.figure(figsize=(6.9, 3.30))
    axes = [fig.add_axes([0.058, 0.220, 0.345, 0.700]),
            fig.add_axes([0.478, 0.220, 0.345, 0.700])]
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
        ax.text(0.5, -0.290, f"非对角元均方根 {rms:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.2, color=fs.MUTED)

    cax = fig.add_axes([0.848, 0.220, 0.022, 0.700])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(0.6)
    cb.set_label("互相关 $C_{ij}$", fontsize=8.6, labelpad=6)
    return fs.save(fig, "practice", "ch03", "barlow-xcorr")


# ---------- 图 3-3 ----------
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
    ax.text(1.62, 1.28, "1σ", fontsize=7.8, color=fs.ACCENT, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.2))

    # 右：1σ 圆（两条主轴等长），配两条短的轴指示
    ax = panels[1]
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(th), np.sin(th), color=fs.GREEN, lw=1.4, ls=(0, (4, 3)), zorder=4)
    for d in ((0.88, 0.0), (0.0, 0.88)):
        ax.annotate("", xy=d, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=fs.GREEN, lw=1.0,
                                    shrinkA=0, shrinkB=0, mutation_scale=7), zorder=5)
    ax.text(0.62, 1.20, "1σ", fontsize=7.8, color=fs.GREEN, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.2))

    notes = [("方差挤在一条斜线上", fs.ACCENT), ("各方向方差为 1" + chr(10) + "方向互不相关", fs.GREEN)]
    for ax, (txt, col) in zip(panels, notes):
        ax.text(0.10, 0.955, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.0, color=col, zorder=8, linespacing=1.45,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.8))
    return fs.save(fig, "practice", "ch03", "whitening")


# ---------- 图 3-4 ----------
def fig_byol_grad():
    fig = plt.figure(figsize=(7.15, 3.30))
    ax = fig.add_axes([0.010, 0.015, 0.560, 0.975])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    bh, fs_b = 0.105, 8.2
    y_on, y_tg = 0.815, 0.345

    # 在线分支
    fs.box(ax, 0.015, y_on - bh / 2, 0.126, bh, "视图 1", fc="#FFFFFF", ec=fs.ACCENT, fs=fs_b, lw=1.1)
    fs.box(ax, 0.175, y_on - bh / 2, 0.146, bh, "编码器\nfθ", fc="#F6F8FB", ec=fs.ACCENT, fs=fs_b, lw=1.2)
    fs.box(ax, 0.355, y_on - bh / 2, 0.146, bh, "投影头\ngθ（BN）", fc="#F6F8FB", ec=fs.ACCENT, fs=7.8, lw=1.2)
    fs.box(ax, 0.535, y_on - bh / 2, 0.136, bh, "预测器\nqθ", fc="#FDF1E9", ec=fs.ACCENT, fs=fs_b, lw=1.2)
    for x0, x1 in ((0.142, 0.173), (0.322, 0.353), (0.502, 0.533)):
        fs.arrow(ax, (x0, y_on), (x1, y_on), color=fs.MUTED, lw=1.1)

    # 目标分支
    fs.box(ax, 0.015, y_tg - bh / 2, 0.126, bh, "视图 2", fc="#FFFFFF", ec=fs.BLUE, fs=fs_b, lw=1.1)
    fs.box(ax, 0.175, y_tg - bh / 2, 0.146, bh, "编码器\nfξ", fc="#F6F8FB", ec=fs.BLUE, fs=fs_b, lw=1.2)
    fs.box(ax, 0.355, y_tg - bh / 2, 0.146, bh, "投影头\ngξ（BN）", fc="#F6F8FB", ec=fs.BLUE, fs=7.8, lw=1.2)
    for x0, x1 in ((0.142, 0.173), (0.322, 0.353)):
        fs.arrow(ax, (x0, y_tg), (x1, y_tg), color=fs.MUTED, lw=1.1)

    # 损失
    fs.box(ax, 0.795, 0.400, 0.185, 0.150, "", fc="#FFFFFF", ec=fs.INK, lw=1.3)
    fs.label(ax, 0.8875, 0.505, "损失", fs=8.8, weight="bold")
    fs.label(ax, 0.8875, 0.443, "L = ‖p − z′‖²", fs=8.4)

    fs.arrow(ax, (0.673, y_on - 0.020), (0.793, 0.558), color=fs.ACCENT, lw=1.3)
    fs.label(ax, 0.726, 0.744, "p", fs=9.4, color=fs.ACCENT, weight="bold")
    mid = stopgrad_arrow(ax, (0.503, y_tg + 0.006), (0.793, 0.424), frac=0.56,
                         color=fs.BLUE, lw=1.3)
    fs.label(ax, 0.512, 0.268, "z′", fs=9.4, color=fs.BLUE, weight="bold")
    fs.label(ax, mid[0], mid[1] - 0.062, "停梯度", fs=7.6, color=fs.BLUE)

    # 反向通路：一条从右端指回的虚线箭头
    fs.arrow(ax, (0.690, 0.700), (0.150, 0.700), color=fs.GREEN, lw=1.4,
             ls=(0, (4.5, 3)), ms=9)
    fs.label(ax, 0.420, 0.662, "梯度沿虚线回流，只有这一条路", fs=7.9, color=fs.GREEN)

    # 目标分支：没有反向箭头
    fs.label(ax, 0.355, 0.205, "× 这一支没有反向箭头，ξ 回不去",
             fs=7.6, color=fs.ACCENT, ha="left")

    # 底部：BN 的作用
    fs.label(ax, 0.050, 0.128, "投影头的输出要过批归一化，μ 与 σ 由同一批样本算出，",
             fs=7.7, color=fs.PURPLE, ha="left")
    fs.label(ax, 0.050, 0.070, "一个样本的输出因此含同批其他样本的信息",
             fs=7.7, color=fs.PURPLE, ha="left")
    fs.label(ax, 0.040, 0.960, "(a) 反向通路", fs=8.6, color=fs.INK, weight="bold", ha="left")

    # 右图：批大小曲线
    bs = np.array([64, 128, 256, 512, 1024, 2048, 4096], float)
    byol = [59.7, 69.6, 71.8, 72.2, 72.2, 72.4, 72.5]
    simclr = [59.2, 63.6, 64.3, 66.5, 67.4, 67.8, 67.9]
    ax2 = fig.add_axes([0.700, 0.190, 0.290, 0.700])
    ax2.axvspan(256, 4096, color=fs.LIGHT, zorder=0)
    ax2.plot(bs, byol, "o-", color=fs.ACCENT, lw=1.5, ms=4.0, label="BYOL")
    ax2.plot(bs, simclr, "s--", color=fs.BLUE, lw=1.3, ms=3.6, label="SimCLR（复现）")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([64, 256, 1024, 4096])
    ax2.set_xticklabels(["64", "256", "1024", "4096"])
    ax2.set_xlim(50, 6500); ax2.set_ylim(56, 76)
    ax2.set_yticks([60, 65, 70, 75])
    ax2.set_xlabel("批大小", fontsize=8.8, labelpad=2)
    ax2.set_ylabel("线性评估 top-1（%）", fontsize=8.8, labelpad=3)
    ax2.tick_params(labelsize=8, length=2)
    fs.clean(ax2)
    ax2.legend(loc="lower right", fontsize=7.6, handlelength=1.5, borderpad=0.15)
    ax2.annotate("64 掉到 59.7", xy=(70, 60.2), xytext=(108, 57.4), fontsize=7.4,
                 color=fs.ACCENT, ha="left",
                 arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8))
    ax2.annotate("256 以上几乎水平", xy=(1500, 72.35), xytext=(140, 74.0), fontsize=7.4,
                 color=fs.MUTED,
                 arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.8))
    ax2.text(0.0, 1.045, "(b) 批大小与精度", transform=ax2.transAxes, fontsize=8.6,
             color=fs.INK, weight="bold", ha="left")
    return fs.save(fig, "practice", "ch03", "byol-grad")


# ---------- 图 3-5 ----------
def _vicreg_demo(use_var=True, use_cov=True, steps=8000, lr=0.02, seed=5):
    """两维嵌入上的最小演示：8 维输入经线性映射到 2 维，四条项组合分别优化。

    三个分支共用同一批输入 X 与两套固定扰动（模拟两个增强视图）。
    方差项与协方差项按 VICReg 的写法，协方差项权重取 25（原论文为 1），
    目的是让 2 维、几百步的演示也能看到收敛。
    """
    rng = np.random.default_rng(seed)
    N, K, D = 400, 8, 2
    X = rng.normal(size=(N, K))
    X[:, 1] = 0.9 * X[:, 0] + 0.43 * rng.normal(size=N)      # 输入自身带强相关
    X1 = X + 0.45 * rng.normal(size=(N, K))
    X2 = X + 0.45 * rng.normal(size=(N, K))

    def loss(W):
        Z1, Z2 = X1 @ W, X2 @ W
        inv = ((Z1 - Z2) ** 2).sum(1).mean()
        var = (np.maximum(0.0, 1.0 - Z1.std(0)).mean()
               + np.maximum(0.0, 1.0 - Z2.std(0)).mean())
        c = 0.0
        for Z in (Z1, Z2):
            C = np.cov(Z.T, bias=False)
            c += ((C - np.diag(np.diag(C))) ** 2).sum() / D
        return 25.0 * inv + (25.0 * var if use_var else 0.0) + (25.0 * c if use_cov else 0.0)

    W = (rng.normal(size=(K, D)) * 0.3) if seed == 5 else np.zeros((K, D))
    W = np.random.default_rng(seed).normal(size=(K, D)) * 0.3
    m = np.zeros_like(W); v = np.zeros_like(W); h = 1e-6
    for t in range(1, steps + 1):
        g = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + h; a = loss(W)
            W[idx] = old - h; b = loss(W)
            W[idx] = old
            g[idx] = (a - b) / (2 * h)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    Z1, Z2 = X1 @ W, X2 @ W
    C = np.cov(Z1.T, bias=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    return Z1, Z2, ev


def fig_regularizers():
    cases = [("只保留不变项", False, False), ("加上方差项", True, False),
             ("三项齐全", True, True)]
    out = [_vicreg_demo(use_var=v, use_cov=c) for _, v, c in cases]

    fig = plt.figure(figsize=(6.7, 4.55))
    pos = [0.070, 0.565, 0.380, 0.380],
    boxes = [(0.070, 0.560, 0.380, 0.360), (0.560, 0.560, 0.380, 0.360),
             (0.070, 0.090, 0.380, 0.360), (0.560, 0.090, 0.380, 0.360)]
    tags = ("(a) 只保留不变项", "(b) 加上方差项", "(c) 三项齐全")
    lim = 3.4
    for k in range(3):
        Z1, Z2, ev = out[k]
        ax = fig.add_axes(boxes[k])
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        std = Z1.std(0)
        rho = 0.0 if std.min() < 1e-6 else np.corrcoef(Z1.T)[0, 1]
        if std.max() < 0.05:
            ax.scatter(Z1[:, 0], Z1[:, 1], s=6, color=fs.FAINT, alpha=0.18,
                       linewidths=0, zorder=2)
            ax.scatter(Z1[:, 0], Z1[:, 1], s=16, color=fs.ACCENT, alpha=0.9,
                       linewidths=0, zorder=3)
            ax.annotate("所有点重合于原点", xy=(0, 0), xytext=(0.45, 1.55),
                        fontsize=7.6, color=fs.ACCENT,
                        arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8))
        else:
            ax.scatter(Z1[:, 0], Z1[:, 1], s=7, color=fs.BLUE, alpha=0.40,
                       linewidths=0, zorder=3)
        ax.axhline(0, color=fs.FAINT, lw=0.6, zorder=1)
        ax.axvline(0, color=fs.FAINT, lw=0.6, zorder=1)
        fs.clean(ax, spines=("left", "bottom"))
        ax.tick_params(labelsize=7, length=2)
        ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
        ax.set_title(tags[k], fontsize=8.8, pad=4)
        ax.set_xlabel("第 1 维", fontsize=7.8, labelpad=1)
        ax.set_ylabel("第 2 维", fontsize=7.8, labelpad=1)
        rank = float(ev.sum() ** 2 / (ev ** 2).sum()) if std.min() > 1e-6 else float("nan")
        rank_txt = "—" if std.min() < 1e-6 else "%.2f" % rank
        note = "标准差 %.2f / %.2f，相关 %.2f，有效秩 %s" % (
            std[0], std[1], rho, rank_txt)
        ax.text(0.03, 0.96, note, transform=ax.transAxes, fontsize=6.9,
                color=fs.MUTED, ha="left", va="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82))
        print("  regularizers %s：%s" % (tags[k], note))

    # (d) 特征值谱
    ax = fig.add_axes(boxes[3])
    wid = 0.26
    for k in range(3):
        ev = out[k][2]
        for t in range(2):
            ax.bar(k + (t - 0.5) * wid, ev[t], width=wid * 0.92,
                   color=(fs.ACCENT if t == 0 else fs.BLUE), alpha=0.85, zorder=3)
        lab = "—" if ev.sum() < 1e-6 else "%.2f" % float(ev.sum() ** 2 / (ev ** 2).sum())
        ax.text(k, -0.55, "有效秩 " + lab, ha="center", va="top",
                fontsize=7.2, color=fs.MUTED)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["(a)", "(b)", "(c)"], fontsize=8)
    ax.set_ylim(0, 2.4); ax.set_xlim(-0.55, 2.55)
    ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=7, length=2)
    ax.set_ylabel("协方差特征值", fontsize=7.8, labelpad=2)
    fs.clean(ax)
    ax.set_title("(d) 协方差谱", fontsize=8.8, pad=4)
    ax.bar(0.42, 0, color=fs.ACCENT, label="第 1 主方向")
    ax.bar(0.42, 0, color=fs.BLUE, label="第 2 主方向")
    ax.legend(loc="upper left", fontsize=7.0, handlelength=1.0, borderpad=0.2)
    return fs.save(fig, "practice", "ch03", "regularizers")


if __name__ == "__main__":
    fig_byol()
    fig_barlow()
    fig_whiten()
    fig_byol_grad()
    fig_regularizers()
