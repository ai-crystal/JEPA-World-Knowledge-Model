# -*- coding: utf-8 -*-
"""第 1 章插图：一维能量地形、二维能量等高线与梯度下降轨迹、L1 与 L2 的损失与梯度。

数值都可以重跑复现，正文里引用的谷底能量、均值点能量、能量比、
离群维度的损失份额都取自本脚本的打印输出。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

# ---- 两个“合理未来”的示意能量函数 --------------------------------------
M1, S1, W1 = -1.20, 0.40, 0.60      # 拿起杯子
M2, S2, W2 = 1.12, 0.44, 0.40       # 推开杯子
NORM = 1.0 / np.sqrt(2 * np.pi)
DIM = 1408                          # V-JEPA 2 的 ViT-g 特征维度


def dens(x):
    p1 = W1 * NORM / S1 * np.exp(-(x - M1) ** 2 / (2 * S1 ** 2))
    p2 = W2 * NORM / S2 * np.exp(-(x - M2) ** 2 / (2 * S2 ** 2))
    return p1 + p2


def energy1(x):
    """E(x) = -log p(x)，把密度换成能量。"""
    return -np.log(dens(x))


def fig_landscape_1d():
    x = np.linspace(-2.35, 2.35, 1600)
    e = energy1(x)
    e0 = e.min()
    e = e - e0

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.fill_between(x, -0.2, e, color="#F6EDE4", zorder=1)
    ax.plot(x, e, color=fs.INK, lw=1.5, zorder=3)

    # 两个谷底
    for m, lab, c, dy in ((M1, "拿起杯子", fs.BLUE, 0.95), (M2, "推开杯子", fs.GREEN, 0.95)):
        em = energy1(np.array([m]))[0] - e0
        ax.plot([m], [em], marker="o", ms=6.5, color=c, zorder=5)
        ax.annotate(lab, xy=(m, em + 0.06), xytext=(m, em + dy),
                    ha="center", va="center", fontsize=10, color=c, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.9,
                                    shrinkA=2, shrinkB=4), zorder=6)

    # 回归解：条件均值，落在两谷之间的坡上
    mu = W1 * M1 + W2 * M2
    emu = energy1(np.array([mu]))[0] - e0
    ax.plot([mu, mu], [0, emu], color=fs.ACCENT, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.plot([mu], [emu], marker="X", ms=9, color=fs.ACCENT, zorder=7)
    ax.text(mu + 0.14, 1.42, "L2 回归\n给出的未来", ha="left", va="center",
            fontsize=9.6, color=fs.ACCENT, linespacing=1.5, zorder=8)

    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-0.2, 4.25)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("能量 $E(y \\mid x)$", fontsize=9.6, color=fs.MUTED)
    fs.clean(ax, spines=())
    ax.annotate("", xy=(2.33, -0.2), xytext=(-2.33, -0.2),
                arrowprops=dict(arrowstyle="-|>", color=fs.FAINT, lw=0.9), zorder=2)
    ax.set_xlabel("未来状态 $y$ 的一个坐标（示意投影）", fontsize=9.6, color=fs.MUTED,
                  labelpad=8)

    eb = energy1(np.array([M1, M2, mu])) - e0
    h = 1e-6
    sc = (dens(h) - dens(-h)) / (2 * h) / dens(0.0)      # x=0 处的得分
    print("[1-1] x=0 处的得分 %.3f，即能量的一阶导数 %.3f" % (sc, -sc))
    mid = (x > M1) & (x < M2)
    ip = int(np.argmax(np.where(mid, e, -1.0)))
    print("[1-1] 两谷之间的墙顶 x=%.2f E=%.3f" % (x[ip], e[ip]))
    print("[1-1] 谷底能量 %.3f / %.3f, 谷底横坐标 %.2f / %.2f, 概率比 %.2f"
          % (eb[0], eb[1], M1, M2, np.exp(eb[1] - eb[0])))
    print("[1-1] 均值点 x=%.3f E=%.3f，是较高谷底的 %.2f 倍"
          % (mu, eb[2], eb[2] / eb[1]))
    return fs.save(fig, "practice", "ch01", "landscape-1d")


def _gauss2(x, y, cx, cy, sa, sb, th, w):
    """各向异性高斯：沿旋转后的两个主轴 sa、sb 展开。"""
    ca, sn = np.cos(th), np.sin(th)
    u = (x - cx) * ca + (y - cy) * sn
    v = -(x - cx) * sn + (y - cy) * ca
    return w * np.exp(-(u ** 2 / (2 * sa ** 2) + v ** 2 / (2 * sb ** 2)))


def energy2(x, y):
    p1 = _gauss2(x, y, M1, 0.60, 0.55, 0.35, 0.40, W1)
    p2 = _gauss2(x, y, 1.05, -0.65, 0.85, 0.42, -0.50, W2)
    return -np.log(p1 + p2 + 1e-12)


def grad2(x, y, h=1e-4):
    gx = (energy2(x + h, y) - energy2(x - h, y)) / (2 * h)
    gy = (energy2(x, y + h) - energy2(x, y - h)) / (2 * h)
    return gx, gy


def descend(x0, y0, lr=0.03, n=420):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(n):
        gx, gy = grad2(x, y)
        gn = np.hypot(gx, gy)
        if gn > 5.0:                      # 限制步长，避免一步飞出画面
            gx, gy = gx / gn * 5.0, gy / gn * 5.0
        x, y = x - lr * gx, y - lr * gy
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def fig_contour():
    gx, gy = np.meshgrid(np.linspace(-3.15, 3.15, 620), np.linspace(-2.35, 2.05, 450))
    z = energy2(gx, gy)

    fig, ax = plt.subplots(figsize=(6.6, 3.95))
    lv = np.linspace(0.0, 4.40, 13)
    cf = ax.contourf(gx, gy, z, levels=lv, cmap="Blues_r", alpha=0.92,
                     extend="max", zorder=1)
    ax.contour(gx, gy, z, levels=lv, colors="#8496AD", linewidths=0.45, zorder=2)

    xs, ys = descend(2.30, 1.80)
    xs2, ys2 = descend(-2.45, 1.70)

    # 另一条从左侧出发的轨迹（淡）
    ax.plot(xs2, ys2, color=fs.MUTED, lw=1.1, ls=(0, (3, 2)), zorder=4, alpha=0.9)
    ax.plot([xs2[0]], [ys2[0]], marker="o", ms=6, mfc="white", mec=fs.MUTED,
            mew=1.3, zorder=8)

    # 主轨迹 + 沿路箭头（按弧长均分）
    ax.plot(xs, ys, color=fs.ACCENT, lw=1.9, zorder=6)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    acc = np.concatenate([[0], np.cumsum(seg)])
    for frac in (0.14, 0.34, 0.55, 0.76):
        i = int(np.searchsorted(acc, frac * acc[-1]))
        i = min(i, len(xs) - 8)
        ax.annotate("", xy=(xs[i + 6], ys[i + 6]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.4,
                                    shrinkA=0, shrinkB=0), zorder=7)

    # 起点与终点
    ax.plot([xs[0]], [ys[0]], marker="o", ms=7.5, mfc="white", mec=fs.ACCENT,
            mew=1.7, zorder=8)
    ax.annotate("起点", xy=(xs[0], ys[0]), xytext=(xs[0] - 0.10, ys[0] + 0.40),
                ha="center", fontsize=9.4, color=fs.ACCENT,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.2),
                arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8, shrinkB=4),
                zorder=9)
    ax.plot([xs[-1]], [ys[-1]], marker="o", ms=6.5, color=fs.ACCENT, zorder=8)
    ax.annotate("收敛到谷底", xy=(xs[-1], ys[-1]), xytext=(xs[-1] + 0.28, ys[-1] - 0.72),
                ha="center", fontsize=9.4, color=fs.ACCENT,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.2),
                arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8, shrinkB=5),
                zorder=9)

    # 另一个谷
    ax.plot([M1], [0.60], marker="o", ms=6.5, color=fs.BLUE, zorder=8)
    ax.annotate("另一个低能量解", xy=(M1, 0.60), xytext=(M1 + 0.10, 1.72),
                ha="center", fontsize=9.4, color=fs.BLUE,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.2),
                arrowprops=dict(arrowstyle="-", color=fs.BLUE, lw=0.8, shrinkB=5),
                zorder=9)

    ax.set_xlim(-3.15, 3.15)
    ax.set_ylim(-2.35, 2.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$y_1$", fontsize=9.6, color=fs.MUTED)
    ax.set_ylabel("$y_2$", fontsize=9.6, color=fs.MUTED)
    fs.clean(ax, spines=())
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.015, ticks=[1.5, 3.0, 4.0])
    cb.set_label("能量", fontsize=9, color=fs.MUTED, rotation=270, labelpad=13)
    cb.ax.tick_params(labelsize=8, colors=fs.MUTED)
    cb.outline.set_visible(False)
    print("[1-2] 主轨迹 (%.2f, %.2f) -> (%.3f, %.3f)，侧轨迹 (%.2f, %.2f) -> (%.3f, %.3f)"
          % (xs[0], ys[0], xs[-1], ys[-1], xs2[0], ys2[0], xs2[-1], ys2[-1]))
    return fs.save(fig, "practice", "ch01", "contour-descent")


def fig_l1_l2():
    r = np.linspace(-3.0, 3.0, 1201)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15))

    ax = axes[0]
    ax.plot(r, r ** 2, color=fs.ACCENT, lw=1.8, zorder=3, label="L2：残差的平方")
    ax.plot(r, np.abs(r), color=fs.BLUE, lw=1.8, zorder=3, label="L1：残差的绝对值")
    ax.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, handlelength=1.6, borderpad=0.5, labelspacing=0.5,
              fontsize=9.2)
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-0.3, 9.6)
    ax.set_xlabel("某一维的残差 $r$", fontsize=9.4, color=fs.MUTED)
    ax.set_ylabel("该维对损失的贡献", fontsize=9.4, color=fs.MUTED)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([0, 3, 6, 9])
    fs.clean(ax)
    ax.axhline(0, color=fs.FAINT, lw=0.7, zorder=1)

    ax = axes[1]
    ax.plot(r, 2 * np.abs(r), color=fs.ACCENT, lw=1.8, zorder=3, label="L2：梯度 $2r$")
    ax.plot(r, np.ones_like(r), color=fs.BLUE, lw=1.8, zorder=3, label=r"L1：梯度 $\pm 1$")
    ax.fill_between([-0.05, 0.05], -1, 1, color=fs.BLUE, alpha=0.30, lw=0, zorder=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.95, handlelength=1.6, borderpad=0.5,
              labelspacing=0.5, fontsize=9.2)
    ax.annotate("零点处 L1 的梯度\n是区间 $[-1,1]$", xy=(0.0, 1.06), xytext=(0.0, 3.35),
                fontsize=8.9, color=fs.BLUE, ha="center", va="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=0.8,
                                shrinkA=4, shrinkB=2))
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-0.3, 6.4)
    ax.set_xlabel("某一维的残差 $r$", fontsize=9.4, color=fs.MUTED)
    ax.set_ylabel("梯度幅度", fontsize=9.4, color=fs.MUTED)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([0, 2, 4, 6])
    fs.clean(ax)
    ax.axhline(0, color=fs.FAINT, lw=0.7, zorder=1)

    fig.subplots_adjust(wspace=0.28)

    # 离群维度在总损失里的份额：一维残差是其余各维的十倍
    oth = DIM - 1
    share_l2 = 10.0 ** 2 / (10.0 ** 2 + oth * 1.0)
    share_l1 = 10.0 / (10.0 + oth * 1.0)
    print("[1-3] %d 维，单维残差 10、其余各维 1：L2 份额 %.2f%%，L1 份额 %.2f%%，相差 %.1f 倍"
          % (DIM, share_l2 * 100, share_l1 * 100, share_l2 / share_l1))
    return fs.save(fig, "practice", "ch01", "l1-vs-l2")


# ---- 图 1-4：沿一条路径切片，顺带看一维投影 -----------------------------
SA = np.array([-1.20, 0.55])        # 候选甲
SB = np.array([1.12, -0.60])        # 候选乙


def _g2(x, y, cx, cy, sx, sy, th, w):
    """二维各向异性高斯，th 是主轴相对横轴的旋转角。"""
    ca, sn = np.cos(th), np.sin(th)
    u = (x - cx) * ca + (y - cy) * sn
    v = -(x - cx) * sn + (y - cy) * ca
    return w / (2 * np.pi * sx * sy) * np.exp(-(u ** 2 / (2 * sx ** 2) + v ** 2 / (2 * sy ** 2)))


def energy_slice(x, y):
    """两个候选各是一小团，能量取负对数密度。"""
    p = _g2(x, y, SA[0], SA[1], 0.40, 0.30, 0.35, 0.60) \
        + _g2(x, y, SB[0], SB[1], 0.44, 0.33, -0.40, 0.40)
    return -np.log(p + 1e-14)


def path_profile(n=1201):
    t = np.linspace(0.0, 1.0, n)
    pts = SA[None, :] + t[:, None] * (SB - SA)[None, :]
    e = np.array([energy_slice(p[0], p[1]) for p in pts])
    return t, pts, e - e.min()


def proj_envelope(nx=520, ny=760):
    """把窗口内的点按横坐标分箱，取每个箱里的最低能量。"""
    xs = np.linspace(-1.55, 1.45, nx)
    ys = np.linspace(-1.90, 1.90, ny)
    gx, gy = np.meshgrid(xs, ys)
    z = energy_slice(gx, gy)
    return xs, z.min(axis=0)


def fig_energy_slice():
    t, pts, e = path_profile()
    xs, env = proj_envelope()
    env = env - env.min()

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.45),
                             gridspec_kw=dict(width_ratios=[1.0, 1.18], wspace=0.30))

    # ---- (a) 二维等高线 + 路径
    ax = axes[0]
    gx, gy = np.meshgrid(np.linspace(-1.90, 1.80, 620), np.linspace(-1.90, 1.80, 620))
    z = energy_slice(gx, gy)
    lv = np.linspace(0.0, 7.6, 16)
    cf = ax.contourf(gx, gy, z, levels=lv, cmap="Blues_r", alpha=0.92, extend="max", zorder=1)
    ax.contour(gx, gy, z, levels=lv, colors="#8496AD", linewidths=0.4, zorder=2)

    ax.plot([SA[0], SB[0]], [SA[1], SB[1]], color=fs.ACCENT, lw=1.6, zorder=5)
    tick = np.linspace(0, 1, 13)
    ax.scatter(SA[0] + tick * (SB[0] - SA[0]), SA[1] + tick * (SB[1] - SA[1]),
               s=7, color="white", edgecolor=fs.ACCENT, linewidth=0.6, zorder=6)

    for p, lab, dx, dy in ((SA, "候选甲", -0.16, 0.30), (SB, "候选乙", 0.14, -0.30)):
        ax.plot([p[0]], [p[1]], marker="o", ms=6.2, color=fs.ACCENT, zorder=7)
        ax.text(p[0] + dx, p[1] + dy, lab, fontsize=9.2, color=fs.ACCENT,
                ha="center", va="center", zorder=8,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.4))

    i = int(np.argmax(e))
    ax.plot([pts[i, 0]], [pts[i, 1]], marker="D", ms=5.0, color="white",
            mec=fs.INK, mew=0.9, zorder=8)
    ax.annotate("墙顶", xy=(pts[i, 0], pts[i, 1]), xytext=(pts[i, 0] - 0.05, pts[i, 1] + 0.62),
                ha="center", fontsize=9.0, color=fs.INK, zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.4),
                arrowprops=dict(arrowstyle="-", color=fs.INK, lw=0.8, shrinkB=4))

    ax.set_xlim(-1.90, 1.80)
    ax.set_ylim(-1.90, 1.80)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$y_1$", fontsize=9.4, color=fs.MUTED)
    ax.set_ylabel("$y_2$", fontsize=9.4, color=fs.MUTED)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=10.0,
            color=fs.INK, va="top", ha="left", weight="bold", zorder=10)
    fs.clean(ax, spines=())
    cb = fig.colorbar(cf, ax=ax, shrink=0.78, pad=0.03, ticks=[2.4, 4.8, 7.2])
    cb.ax.tick_params(labelsize=7.8, colors=fs.MUTED)
    cb.outline.set_visible(False)

    # ---- (b) 剖面与投影下包络
    ax = axes[1]
    m = (xs >= SA[0] - 0.10) & (xs <= SB[0] + 0.10)
    ax.plot(xs[m], env[m], color=fs.BLUE, lw=1.3, ls=(0, (4, 2.5)), zorder=3,
            label="一维投影：下包络")
    ax.plot(pts[:, 0], e, color=fs.ACCENT, lw=1.9, zorder=4, label="沿路径切片")

    hb = e[i]
    he = env[m].max() if np.isfinite(env[m]).any() else np.nan
    ax.plot([pts[i, 0]], [hb], marker="o", ms=5.6, color=fs.ACCENT, zorder=6)
    j = int(np.nanargmax(np.where(m, env, np.nan)))
    ax.plot([xs[j]], [env[j]], marker="o", ms=5.6, color=fs.BLUE, zorder=6)

    ax.annotate("%.2f" % hb, xy=(pts[i, 0] + 0.04, hb), xytext=(0.56, hb), fontsize=9.0,
                color=fs.ACCENT, ha="left", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8, shrinkB=6))
    ax.annotate("%.2f" % he, xy=(xs[j] - 0.04, env[j]), xytext=(-0.46, he), fontsize=9.0,
                color=fs.BLUE, ha="right", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", color=fs.BLUE, lw=0.8, shrinkB=6))
    ax.annotate("", xy=(1.05, hb), xytext=(1.05, he),
                arrowprops=dict(arrowstyle="<->", color=fs.MUTED, lw=0.8,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text(1.11, (hb + he) / 2.0, "%.2f" % (hb - he), fontsize=8.6,
            color=fs.MUTED, ha="left", va="center", zorder=8)

    ax.set_xlim(SA[0] - 0.12, SB[0] + 0.40)
    ax.set_ylim(-0.35, 6.55)
    ax.set_xlabel("沿路径的横坐标 $y_1$", fontsize=9.2, color=fs.MUTED)
    ax.set_ylabel("能量（纳特）", fontsize=9.2, color=fs.MUTED)
    ax.set_yticks([0, 2, 4, 6])
    ax.legend(loc="upper right", bbox_to_anchor=(1.00, 1.02), frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.95,
              handlelength=1.7, fontsize=8.8, labelspacing=0.4)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, fontsize=10.0,
            color=fs.INK, va="top", ha="left", weight="bold", zorder=10)
    fs.clean(ax, grid=True)

    print("[1-4] 剖面墙高 %.3f 在 t=%.3f（横坐标 %.2f），两谷底 %.3f / %.3f"
          % (hb, t[i], pts[i, 0], e[0], e[-1]))
    print("[1-4] 投影下包络峰值 %.3f，低估 %.3f 纳特，概率上 %.2f 倍"
          % (he, hb - he, np.exp(hb - he)))
    return fs.save(fig, "practice", "ch01", "energy-slice")


# ---- 图 1-5：三种损失的形状与梯度 ---------------------------------------
def fig_loss_types():
    d = np.linspace(-0.5, 8.0, 1400)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.92),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.05], wspace=0.34))

    KS = (8, 64, 256)
    cols = (fs.BLUE, fs.GREEN, fs.PURPLE)

    # (a) 损失随能量差
    ax = axes[0]
    for K, c in zip(KS, cols):
        ax.plot(d, np.log1p((K - 1) * np.exp(-d)), color=c, lw=1.6, zorder=3,
                label="对比式 $K=%d$" % K)
    ax.plot(d, np.maximum(0.0, 2.0 - d), color=fs.ACCENT, lw=1.6, zorder=3, label="hinge $m=2$")
    ax.set_xlim(-0.5, 8.0)
    ax.set_ylim(-0.25, 6.7)
    ax.set_xlabel("能量差 $E_\\mathrm{neg}-E_\\mathrm{pos}$", fontsize=8.8, color=fs.MUTED)
    ax.set_ylabel("损失", fontsize=9.0, color=fs.MUTED)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 2, 4, 6])
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, fontsize=8.4, handlelength=1.5, labelspacing=0.35,
              borderpad=0.4)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=10.0, color=fs.INK,
            va="top", ha="left", weight="bold", zorder=10)
    fs.clean(ax, grid=True)

    # (b) 对能量差的梯度
    ax = axes[1]
    for K, c in zip(KS, cols):
        g = (K - 1) * np.exp(-d) / (1.0 + (K - 1) * np.exp(-d))
        ax.plot(d, g, color=c, lw=1.6, zorder=3)
        ds = np.log(K - 1)
        ax.plot([ds], [0.5], marker="o", ms=4.4, color=c, zorder=5)
    ax.plot(d, np.where(d < 2.0, 1.0, 0.0), color=fs.ACCENT, lw=1.6, zorder=3)
    ax.text(4.10, -0.17, "余量之外梯度为零", fontsize=8.2, color=fs.ACCENT,
            ha="center", va="center", zorder=8)
    ax.set_xlim(-0.5, 8.0)
    ax.set_ylim(-0.32, 1.12)
    ax.set_xlabel("能量差 $E_\\mathrm{neg}-E_\\mathrm{pos}$", fontsize=8.8, color=fs.MUTED)
    ax.set_ylabel("梯度幅度", fontsize=9.0, color=fs.MUTED)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 0.5, 1.0])
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, fontsize=10.0, color=fs.INK,
            va="top", ha="left", weight="bold", zorder=10)
    fs.clean(ax, grid=True)

    # (c) 负样本数与梯度峰值所在位置
    ax = axes[2]
    K = np.logspace(np.log10(2.0), 6.0, 400)
    ax.plot(K, np.log(K - 1), color=fs.INK, lw=1.6, zorder=3)
    for Kv, c, lab in ((256, fs.GREEN, "$K=256$\n5.54"), (65536, fs.PURPLE, "$K=65536$\n11.09")):
        ax.plot([Kv], [np.log(Kv - 1)], marker="o", ms=5.2, color=c, zorder=5)
        ax.annotate(lab, xy=(Kv, np.log(Kv - 1)), xytext=(Kv * 0.075, np.log(Kv - 1) + 1.45),
                    fontsize=8.4, color=c, ha="center", va="center", linespacing=1.5,
                    zorder=8, arrowprops=dict(arrowstyle="-", color=c, lw=0.7, shrinkB=4))
    ax.set_xscale("log")
    ax.set_xlim(2.0, 1e6)
    ax.set_ylim(0.0, 13.2)
    ax.set_xlabel("候选总数 $K$", fontsize=8.8, color=fs.MUTED)
    ax.set_ylabel("梯度峰值处的能量差（纳特）", fontsize=8.8, color=fs.MUTED)
    ax.set_yticks([0, 4, 8, 12])
    ax.text(0.02, 0.97, "(c)", transform=ax.transAxes, fontsize=10.0, color=fs.INK,
            va="top", ha="left", weight="bold", zorder=10)
    fs.clean(ax, grid=True)

    for K_ in KS:
        print("[1-5] K=%6d 梯度峰值在能量差 %.3f 纳特，即 %.2f 比特"
              % (K_, np.log(K_ - 1), np.log(K_ - 1) / np.log(2.0)))
    return fs.save(fig, "practice", "ch01", "loss-types")


if __name__ == "__main__":
    fig_landscape_1d()
    fig_contour()
    fig_l1_l2()
    fig_energy_slice()
    fig_loss_types()
