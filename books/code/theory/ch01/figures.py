# -*- coding: utf-8 -*-
"""第 1 章插图：三条选择性压力与一次测量、校准前后的相关系数、结构-几何 2x2 框架、
互近邻的置换零分布、三个度量的名次一致性。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs


def fig_three_pressures():
    fig = plt.figure(figsize=(7.0, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    causes = [
        ("任务通用性", "解的任务越多\n可行解越少", fs.GREEN, 0.700),
        ("容量", "假设空间越大\n越容易盖住最优解", fs.BLUE, 0.460),
        ("简单性偏置", "模型越大\n越偏向简单的拟合", fs.PURPLE, 0.220),
    ]
    for name, sub, col, y in causes:
        fs.box(ax, 0.030, y, 0.225, 0.155, "", fc="#FFFFFF", ec=col, lw=1.2)
        fs.label(ax, 0.056, y + 0.112, name, fs=10.2, weight="bold", color=col, ha="left")
        fs.label(ax, 0.056, y + 0.086, sub, fs=7.4, color=fs.MUTED, ha="left", va="top")

    # 中间：被解释的对象
    fs.box(ax, 0.368, 0.415, 0.190, 0.185, "", fc="#F7F4EE", ec=fs.INK, lw=1.2)
    fs.label(ax, 0.463, 0.545, "共享的", fs=9.4, color=fs.INK)
    fs.label(ax, 0.463, 0.492, "世界结构", fs=9.4, color=fs.INK)
    fs.label(ax, 0.463, 0.438, "同一个解", fs=7.4, color=fs.MUTED)

    # 右侧：实际上被测量的东西
    fs.box(ax, 0.668, 0.415, 0.255, 0.185, "", fc="#FFFFFF", ec=fs.ACCENT, lw=1.3)
    fs.label(ax, 0.7955, 0.545, "测得的", fs=9.4, color=fs.ACCENT)
    fs.label(ax, 0.7955, 0.492, "相似度分数", fs=9.4, color=fs.ACCENT)
    fs.label(ax, 0.7955, 0.438, "CKA / mKNN", fs=7.4, color=fs.MUTED)

    fs.arrow(ax, (0.258, 0.778), (0.365, 0.590), color=fs.GREEN, lw=1.1, rad=-0.10)
    fs.arrow(ax, (0.258, 0.538), (0.365, 0.512), color=fs.BLUE, lw=1.1)
    fs.arrow(ax, (0.258, 0.298), (0.365, 0.435), color=fs.PURPLE, lw=1.1, rad=0.10)
    fs.arrow(ax, (0.560, 0.508), (0.665, 0.508), color=fs.INK, lw=1.2)

    # 下方的测量环节，虚线通向分数
    fs.box(ax, 0.368, 0.105, 0.245, 0.110, "", fc="#FFFFFF", ec=fs.FAINT, lw=1.0, rounded=0.015)
    fs.label(ax, 0.4905, 0.177, "测量环节", fs=8.8, color=fs.MUTED)
    fs.label(ax, 0.4905, 0.133, "宽度、深度、层对取最大", fs=7.4, color=fs.MUTED)
    fs.arrow(ax, (0.615, 0.192), (0.745, 0.412), color=fs.FAINT, lw=1.1,
             ls=(0, (4, 3)), rad=-0.06)
    fs.label(ax, 0.592, 0.268, "伪影通道", fs=7.6, color=fs.MUTED, ha="right")

    fs.label(ax, 0.030, 0.960, "可能的成因", fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax, 0.030, 0.922, "三者并不互斥", fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax, 0.030, 0.045, "三者的共同前提：所有模型都在同一个世界上取样",
             fs=7.6, color=fs.MUTED, ha="left")
    return fs.save(fig, "theory", "ch01", "three-pressures")


def fig_calibrated():
    """横轴是分数与模型能力排名的相关系数，左空心点是校准前，右实心点是校准后。"""
    fig = plt.figure(figsize=(6.6, 3.0))
    ax = fig.add_axes([0.235, 0.20, 0.715, 0.66])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.62, 3.72)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(fs.FAINT)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("与语言模型能力排名的相关系数", fontsize=8.6, color=fs.MUTED, labelpad=5)
    ax.tick_params(axis="x", colors=fs.MUTED, labelsize=8.0, length=3)

    rows = [
        ("线性 CKA", fs.ACCENT, 3, 0.86, 0.45, "0.86", "0.45"),
        ("普罗克鲁斯提斯距离", fs.GOLD, 2, 0.89, 0.39, "0.89", "0.39"),
        ("互近邻 mKNN", fs.BLUE, 1, None, 0.85, None, "0.85"),
        ("CKNNA", fs.TEAL, 0, None, 0.87, None, "0.87"),
    ]

    for name, col, y, before, after, lb, la in rows:
        if y % 2 == 1:
            ax.add_patch(Rectangle((0.0, y - 0.42), 1.0, 0.84,
                                   facecolor="#F8FAFC", edgecolor="none", zorder=0))
        if before is None:
            # 校准前后几乎重合，只标一个点
            ax.plot([after, after], [y - 0.22, y + 0.22], color=col, lw=5, alpha=0.20,
                    solid_capstyle="round", zorder=2)
            ax.scatter([after], [y], s=62, facecolor=col, edgecolor=col, linewidth=1.5, zorder=4)
            fs.label(ax, after, y + 0.30, la, fs=8.4, color=col)
            fs.label(ax, after, y - 0.34, "前后几乎重合", fs=7.4, color=fs.MUTED)
        else:
            ax.plot([after, before], [y, y], color=col, lw=6, alpha=0.20,
                    solid_capstyle="round", zorder=2)
            ax.scatter([before], [y], s=62, facecolor="white", edgecolor=col,
                       linewidth=1.5, zorder=4)
            ax.scatter([after], [y], s=62, facecolor=col, edgecolor=col,
                       linewidth=1.5, zorder=4)
            fs.arrow(ax, (before - 0.02, y + 0.20), (after + 0.02, y + 0.20),
                     color=col, lw=0.9, ms=7)
            fs.label(ax, before, y + 0.42, lb, fs=8.4, color=col)
            fs.label(ax, after, y + 0.42, la, fs=8.4, color=col)
            fs.label(ax, before, y - 0.34, "校准前", fs=7.4, color=fs.MUTED)
            fs.label(ax, after, y - 0.34, "校准后", fs=7.4, color=fs.MUTED)
        fs.label(ax, -0.035, y, name, fs=8.6, color=fs.INK, ha="right")

    return fs.save(fig, "theory", "ch01", "calibrated")


def fig_structure_geometry():
    fig = plt.figure(figsize=(6.7, 4.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    W = 0.378
    LX, GX = 0.150, 0.556
    fs.label(ax, LX + W / 2, 0.955, "局部", fs=10.4, color=fs.INK, weight="bold")
    fs.label(ax, GX + W / 2, 0.955, "全局", fs=10.4, color=fs.INK, weight="bold")

    fs.label(ax, 0.083, 0.730, "关系结构", fs=10.2, color=fs.GREEN, weight="bold", rotation=90)
    fs.label(ax, 0.083, 0.375, "度量几何", fs=10.2, color=fs.ACCENT, weight="bold", rotation=90)

    cells = [
        (LX, 0.655, fs.GREEN, "#F1F7F3",
         "互近邻 mKNN", "数两边近邻集合的重合比例", "204/204 对显著"),
        (GX, 0.655, fs.GREEN, "#F1F7F3",
         "骨架重合 $H_0$", "最小生成树共有边占 $n-1$ 的比例", "204/204 对显著"),
        (LX, 0.300, fs.ACCENT, "#FFF7F2",
         "距离感知 mKNN", "在此基础上追加距离接近的要求", "τ = 1 保留 85.6%"),
        (GX, 0.300, fs.ACCENT, "#FFF7F2",
         "距离感知骨架", "距离按各自第 90 百分位归一化", "τ = 1 保留 83.0%"),
    ]
    for x, y, col, fc, title, sub, num in cells:
        fs.box(ax, x, y, W, 0.150, "", fc=fc, ec=col, lw=1.2)
        fs.label(ax, x + 0.016, y + 0.112, title, fs=9.6, color=col, ha="left", weight="bold")
        fs.label(ax, x + 0.016, y + 0.070, sub, fs=7.4, color=fs.MUTED, ha="left")
        fs.label(ax, x + 0.016, y + 0.030, num, fs=8.0, color=fs.INK, ha="left")

    fs.label(ax, LX, 0.610, "校准后仍然显著，且随能力上升", fs=7.6, color=fs.MUTED, ha="left")
    fs.label(ax, LX, 0.255, "越要求距离一致，对齐衰减越快", fs=7.6, color=fs.MUTED, ha="left")

    fs.arrow(ax, (LX + W / 2, 0.232), (LX + W / 2, 0.178), color=fs.ACCENT, lw=1.0, ms=7)
    fs.label(ax, LX + W / 2, 0.155, "τ = $10^{-2}$ 时只剩 5.8%", fs=7.8, color=fs.ACCENT)
    fs.arrow(ax, (GX + W / 2, 0.232), (GX + W / 2, 0.178), color=fs.ACCENT, lw=1.0, ms=7)
    fs.label(ax, GX + W / 2, 0.155, "τ = $10^{-2}$ 时只剩 5.5%", fs=7.8, color=fs.ACCENT)

    fs.label(ax, LX, 0.078, "最严一档（τ = $10^{-4}$）仍显著的比例：局部 89.7%，全局 40.2%",
             fs=7.9, color=fs.INK, ha="left")
    return fs.save(fig, "theory", "ch01", "structure-geometry")


def fig_null_dist():
    """互近邻在 n=1024、k=10 下的置换零分布，以及观测值 0.16 落在哪里。

    零分布直接由随机高斯表征算出来：X 与 Y 各 1024 行 768 列，互不相关，
    把 Y 的行置换 K=200 次，每次重算两边的 k 近邻集合的重合比例。
    """
    rng = np.random.default_rng(20240)
    n, k, d, K = 1024, 10, 768, 200
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d))

    def knn_sets(P):
        G = P @ P.T
        np.fill_diagonal(G, -np.inf)
        idx = np.argpartition(-G, k, axis=1)[:, :k]
        return [set(row.tolist()) for row in idx]

    NX = knn_sets(X)
    null = []
    for _ in range(K):
        NY = knn_sets(Y[rng.permutation(n)])
        null.append(np.mean([len(NX[i] & NY[i]) / k for i in range(n)]))
    null = np.asarray(null)

    s_obs = 0.16                      # Huh 等人跨模态的最高读数
    mu0 = k / (n - 1)                 # 命题 4.2 的零基线
    comb = np.sort(np.concatenate([null, [s_obs]]))
    tau = comb[int(np.ceil(0.95 * (K + 1))) - 1]
    sd = null.std(ddof=1)
    print("null mean %.5f  sd %.5f  tau %.5f  (obs-mean)/sd %.1f"
          % (null.mean(), sd, tau, (s_obs - null.mean()) / sd))

    fig = plt.figure(figsize=(6.9, 3.65))
    ax1 = fig.add_axes([0.098, 0.605, 0.878, 0.360])
    ax2 = fig.add_axes([0.098, 0.145, 0.878, 0.290])

    # (a) 零分布本体
    ax1.hist(null, bins=26, color=fs.BLUE, alpha=0.28, edgecolor=fs.BLUE, linewidth=0.7,
             zorder=2)
    ax1.axvline(mu0, color=fs.MUTED, lw=1.0, ls=(0, (5, 3)), zorder=3)
    ax1.axvline(tau, color=fs.ACCENT, lw=1.2, zorder=3)
    ax1.set_xlim(0.005, 0.0155)
    ax1.set_ylim(0, 44)
    fs.clean(ax1, spines=("left", "bottom"))
    ax1.set_ylabel("置换次数", fontsize=8.4, color=fs.MUTED, labelpad=4)
    ax1.tick_params(axis="both", labelsize=8.0, colors=fs.MUTED, length=3)
    ax1.set_xticks([0.006, 0.008, 0.010, 0.012, 0.014])
    fs.label(ax1, 0.0055, 41, "(a)", fs=9.2, color=fs.MUTED, ha="left")
    fs.label(ax1, mu0 + 0.00012, 33, "$k/(n-1)=0.0098$", fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax1, tau + 0.00015, 24, "$\\tau_\\alpha$（$\\alpha=0.05$）", fs=8.2,
             color=fs.ACCENT, ha="left")
    fs.label(ax1, 0.0154, 41, "$n=1024$，$k=10$，$K=200$", fs=8.0, color=fs.MUTED, ha="right")
    fs.label(ax1, 0.0154, 12, "观测值 0.16 远在右侧轴外", fs=8.2, color=fs.ACCENT, ha="right")

    # (b) 同一条数轴上的位置关系
    ax2.set_xlim(-0.004, 0.182)
    ax2.set_ylim(-1.05, 1.05)
    ax2.axis("off")
    fs.arrow(ax2, (-0.002, 0.0), (0.181, 0.0), color=fs.FAINT, lw=1.0, ms=7)
    for t in (0.0, 0.05, 0.10, 0.15):
        ax2.plot([t, t], [-0.10, 0.10], color=fs.FAINT, lw=0.9)
        fs.label(ax2, t, -0.30, "%.2f" % t, fs=7.8, color=fs.MUTED)
    ax2.add_patch(Rectangle((null.min(), -0.34), null.max() - null.min(), 0.68,
                            facecolor=fs.BLUE, alpha=0.16, edgecolor="none", zorder=1))
    fs.label(ax2, 0.021, 0.52, "零分布全部挤在这一段", fs=8.2, color=fs.BLUE, ha="left")
    fs.arrow(ax2, (0.021, 0.44), (null.max() + 0.001, 0.10), color=fs.BLUE, lw=0.9, ms=6)
    ax2.plot([s_obs, s_obs], [-0.36, 0.36], color=fs.ACCENT, lw=1.6, zorder=3)
    ax2.scatter([s_obs], [0.0], s=26, facecolor=fs.ACCENT, edgecolor="white",
                linewidth=0.8, zorder=4)
    fs.label(ax2, s_obs - 0.004, 0.62, "观测值 0.16", fs=9.0, color=fs.ACCENT, ha="right")
    fs.label(ax2, s_obs - 0.004, 0.30, "约 %.0f 个零分布标准差之外" % ((s_obs - null.mean()) / sd),
             fs=7.8, color=fs.MUTED, ha="right")
    fs.label(ax2, 0.176, -0.62, "互近邻分数", fs=8.0, color=fs.MUTED, ha="right")
    fs.label(ax2, -0.004, 0.62, "(b)", fs=9.2, color=fs.MUTED, ha="right")
    return fs.save(fig, "theory", "ch01", "null-dist")


def fig_metric_agree():
    """三个度量对同一批 204 对模型给出的名次有多一致。

    数据是模拟的：每一对模型有一个潜在的能力值，三个分数各自与它的相关系数
    按论文报告的校准后数值设定（CKA 0.45、Procrustes 0.39、mKNN 0.85），
    其余为独立噪声。图上标的是样本名次的 Spearman 相关，只表示形状。
    """
    rng = np.random.default_rng(207)
    m = 204
    cap = rng.standard_normal(m)
    cap = (cap - cap.mean()) / cap.std()

    def draw(rho):
        e = rng.standard_normal(m)
        e = (e - e.mean()) / e.std()
        return rho * cap + np.sqrt(1.0 - rho ** 2) * e

    scores = {"CKA": draw(0.45), "Procrustes": draw(0.39), "mKNN": draw(0.85)}
    ranks = {key: np.argsort(np.argsort(v)) + 1.0 for key, v in scores.items()}

    pairs = [("mKNN", "CKA"), ("mKNN", "Procrustes"), ("CKA", "Procrustes")]
    fig = plt.figure(figsize=(7.1, 2.95))
    for i, (a, b) in enumerate(pairs):
        ax = fig.add_axes([0.088 + i * 0.312, 0.175, 0.238, 0.735])
        rho = float(np.corrcoef(ranks[a], ranks[b])[0, 1])
        ax.plot([-8, m + 9], [-8, m + 9], color=fs.FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.scatter(ranks[a], ranks[b], s=7.5, color=fs.PALETTE[i], alpha=0.45,
                   linewidth=0, zorder=2)
        ax.set_xlim(-6, m + 6)
        ax.set_ylim(-6, m + 6)
        ax.set_xticks([1, 100, 204])
        ax.set_yticks([1, 100, 204])
        fs.clean(ax, spines=("left", "bottom"))
        ax.tick_params(labelsize=7.8, colors=fs.MUTED, length=2.5)
        ax.set_xlabel("%s 名次" % a, fontsize=8.2, color=fs.MUTED, labelpad=3)
        ax.set_ylabel("%s 名次" % b, fontsize=8.2, color=fs.MUTED, labelpad=3)
        ax.text(0.955, 0.055, r"$\rho_S=%.2f$" % rho, transform=ax.transAxes,
                fontsize=8.6, color=fs.PALETTE[i], ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="none", alpha=0.88), zorder=6)
        ax.text(0.045, 0.945, "(%s)" % "abc"[i], transform=ax.transAxes,
                fontsize=8.6, color=fs.MUTED, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none", alpha=0.85), zorder=6)
    return fs.save(fig, "theory", "ch01", "metric-agree")


if __name__ == "__main__":
    fig_three_pressures()
    fig_calibrated()
    fig_structure_geometry()
    fig_null_dist()
    fig_metric_agree()
