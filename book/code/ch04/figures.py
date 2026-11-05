# -*- coding: utf-8 -*-
"""第 4 章插图：一维能量地形、二维能量等高线与梯度下降轨迹。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

# ---- 两个“合理未来”的示意能量函数 --------------------------------------
M1, S1, W1 = -1.20, 0.40, 0.60      # 拿起杯子
M2, S2, W2 = 1.12, 0.44, 0.40       # 推开杯子
NORM = 1.0 / np.sqrt(2 * np.pi)


def dens(x):
    p1 = W1 * NORM / S1 * np.exp(-(x - M1) ** 2 / (2 * S1 ** 2))
    p2 = W2 * NORM / S2 * np.exp(-(x - M2) ** 2 / (2 * S2 ** 2))
    return p1 + p2


def energy1(x):
    """E(x) = -log p(x)，把密度换成能量。"""
    return -np.log(dens(x))


def fig_landscape_1d():
    x = np.linspace(-3.2, 3.4, 1600)
    e = energy1(x)
    e0 = e.min()
    e = e - e0

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.fill_between(x, 0, e, color="#F6EDE4", zorder=1)
    ax.plot(x, e, color=fs.INK, lw=1.5, zorder=3)

    # 两个谷底
    for m, lab, c in ((M1, "拿起杯子", fs.BLUE), (M2, "推开杯子", fs.GREEN)):
        em = energy1(np.array([m]))[0] - e0
        ax.plot([m], [em], marker="o", ms=6.5, color=c, zorder=5)
        ax.annotate(lab, xy=(m, em + 0.05), xytext=(m, em + 1.02),
                    ha="center", va="center", fontsize=10, color=c, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.9,
                                    shrinkA=2, shrinkB=4), zorder=6)

    # 回归解：条件均值，落在两谷之间的墙上
    mu = W1 * M1 + W2 * M2
    emu = energy1(np.array([mu]))[0] - e0
    ax.plot([mu, mu], [0, emu], color=fs.ACCENT, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.plot([mu], [emu], marker="X", ms=9, color=fs.ACCENT, zorder=7)
    ax.text(mu + 0.07, 0.55, "L2 回归\n给出的未来", ha="left", va="center",
            fontsize=9.6, color=fs.ACCENT, linespacing=1.5, zorder=8)

    ax.set_xlim(-3.2, 3.4)
    ax.set_ylim(-0.16, 3.30)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("未来状态 $y$ 的一个坐标（示意投影）", fontsize=9.6, color=fs.MUTED)
    ax.set_ylabel("能量 $E(y \\mid x)$", fontsize=9.6, color=fs.MUTED)
    fs.clean(ax, spines=("bottom",))
    ax.annotate("", xy=(3.36, -0.05), xytext=(-3.16, -0.05),
                arrowprops=dict(arrowstyle="-|>", color=fs.FAINT, lw=0.9))

    eb = energy1(np.array([M1, M2, mu])) - e0
    print("[4-1] 谷底能量 %.3f / %.3f, 回归点 x=%.3f E=%.3f, 比深谷高 %.2f 倍"
          % (eb[0], eb[1], mu, eb[2], eb[2] / eb[1]))
    return fs.save(fig, "ch04", "landscape-1d")


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
    gx, gy = np.meshgrid(np.linspace(-2.7, 2.7, 460), np.linspace(-2.4, 2.4, 400))
    z = np.clip(energy2(gx, gy), 0, 4.6)

    fig, ax = plt.subplots(figsize=(6.6, 3.95))
    lv = np.linspace(0.45, 4.55, 13)
    cf = ax.contourf(gx, gy, z, levels=lv, cmap="Blues_r", alpha=0.92, zorder=1)
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
    ax.annotate("起点", xy=(xs[0], ys[0]), xytext=(xs[0] - 0.02, ys[0] + 0.42),
                ha="center", fontsize=9.4, color=fs.ACCENT,
                bbox=dict(fc="white", ec="none", alpha=0.55, pad=1.2),
                arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8, shrinkB=4),
                zorder=9)
    ax.plot([xs[-1]], [ys[-1]], marker="o", ms=6.5, color=fs.ACCENT, zorder=8)
    ax.annotate("收敛到谷底", xy=(xs[-1], ys[-1]), xytext=(xs[-1] + 0.05, ys[-1] - 0.92),
                ha="center", fontsize=9.4, color=fs.ACCENT,
                bbox=dict(fc="white", ec="none", alpha=0.55, pad=1.2),
                arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8, shrinkB=5),
                zorder=9)

    # 另一个谷
    ax.plot([M1], [0.60], marker="o", ms=6.5, color=fs.BLUE, zorder=8)
    ax.annotate("另一个低能量解", xy=(M1, 0.60), xytext=(M1 + 0.12, 1.86),
                ha="center", fontsize=9.4, color=fs.BLUE,
                bbox=dict(fc="white", ec="none", alpha=0.55, pad=1.2),
                arrowprops=dict(arrowstyle="-", color=fs.BLUE, lw=0.8, shrinkB=5),
                zorder=9)

    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$y_1$", fontsize=9.6, color=fs.MUTED)
    ax.set_ylabel("$y_2$", fontsize=9.6, color=fs.MUTED)
    fs.clean(ax, spines=())
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.015, ticks=[1.5, 3.0, 4.5])
    cb.set_label("能量", fontsize=9, color=fs.MUTED)
    cb.ax.tick_params(labelsize=8, colors=fs.MUTED)
    cb.outline.set_visible(False)
    print("[4-2] 主轨迹终点 (%.3f, %.3f), 侧轨迹终点 (%.3f, %.3f)"
          % (xs[-1], ys[-1], xs2[-1], ys2[-1]))
    return fs.save(fig, "ch04", "contour-descent")


if __name__ == "__main__":
    fig_landscape_1d()
    fig_contour()
