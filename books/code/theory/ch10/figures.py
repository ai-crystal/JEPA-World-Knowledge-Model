# -*- coding: utf-8 -*-
"""第 10 章插图：（1）实验设计矩阵；（2）功效曲线；（3）delta 与 rho 的估计流程；（4）报告模板。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats as st
import figstyle as fs


def fig_design_matrix():
    """三行自变量乘四列设置，末行标出每一组预期改动定理里的哪一项。"""
    fig, ax = fs.canvas(7.4, 4.1)
    rows = [
        ("正样本 $\\pi$", ["同图\n两描述", "同条描述\n两次掩码", "同图\n两描述", "同条描述\n两次掩码"]),
        ("干扰（$n_0$）", ["$5$ 条独立\n描述", "$5$", "$5$", "$n_0=2$\n强改写"]),
        ("间隙（$k$ 与簇）", ["两簇\n$k=1$", "同左", "加一条\n桥边", "同左"]),
    ]
    moved = [[False, True, False, True],
             [False, False, False, True],
             [False, False, True, False]]
    kinds = ["自变量", "自变量", "自变量"]
    heads = ["S0 基线", "S1 改 $\\pi$", "S2 改间隙", "S3 改干扰"]
    x0, y0 = 0.245, 0.305
    bw, bh, gx, gy = 0.172, 0.158, 0.010, 0.022
    for j, h in enumerate(heads):
        fs.label(ax, x0 + j * (bw + gx) + bw / 2, y0 + 2 * (bh + gy) + bh + 0.026, h,
                 fs=9.4, color=fs.INK, ha="center", va="bottom")
    for i, (name, vals) in enumerate(rows):
        y = y0 + (2 - i) * (bh + gy)
        fs.label(ax, x0 - 0.032, y + bh / 2 + 0.020, name, fs=9.4, color=fs.INK,
                 ha="right", va="center")
        fs.label(ax, x0 - 0.032, y + bh / 2 - 0.040, kinds[i], fs=7.4, color=fs.FAINT,
                 ha="right", va="center")
        for j, v in enumerate(vals):
            x = x0 + j * (bw + gx)
            if moved[i][j]:
                fc, ec, tc, lw = fs.ACCENT, fs.ACCENT, "white", 1.0
            else:
                fc, ec, tc, lw = "#FFFFFF", "#CBD5E1", fs.MUTED, 0.9
            ax.add_patch(Rectangle((x, y), bw, bh, facecolor=fc, edgecolor=ec, lw=lw, zorder=2))
            ax.text(x + bw / 2, y + bh / 2, v, ha="center", va="center",
                    fontsize=8.2, color=tc, zorder=3, linespacing=1.55)
    # 末行：预期读数
    y = y0 - (bh + gy)
    fs.label(ax, x0 - 0.032, y + bh / 2 + 0.020, "预期读数", fs=9.4, color=fs.ACCENT,
             ha="right", va="center")
    fs.label(ax, x0 - 0.032, y + bh / 2 - 0.040, "读数", fs=7.4, color=fs.FAINT,
             ha="right", va="center")
    reads = ["基线，三个量\n都量一遍", "核失配升，\n$\\delta$ 略降",
             "分母 $\\delta$ 降，\n右端被推远", "$\\rho$ 降，\n下界抬高"]
    for j, t in enumerate(reads):
        x = x0 + j * (bw + gx)
        ax.add_patch(Rectangle((x, y), bw, bh, facecolor=fs.PAPER,
                               edgecolor=fs.ACCENT, lw=0.8, ls=(0, (3, 2)), zorder=2))
        ax.text(x + bw / 2, y + bh / 2, t, ha="center", va="center", fontsize=7.9,
                color=fs.ACCENT, zorder=3, linespacing=1.55)
    ax.add_patch(Rectangle((0.035, 0.045), 0.028, 0.044, facecolor=fs.ACCENT,
                           edgecolor=fs.ACCENT, lw=1.0, zorder=3))
    fs.label(ax, 0.072, 0.067, "这一组动这个量", fs=8.6, color=fs.MUTED, ha="left", va="center")
    ax.add_patch(Rectangle((0.275, 0.045), 0.028, 0.044, facecolor="white",
                           edgecolor="#CBD5E1", lw=0.9, zorder=3))
    fs.label(ax, 0.312, 0.067, "按住不动", fs=8.6, color=fs.MUTED, ha="left", va="center")
    fs.label(ax, 0.470, 0.067, "末行是每一组应当读出来的量，不是结果",
             fs=8.6, color=fs.MUTED, ha="left", va="center")
    return fs.save(fig, "theory", "ch10", "design-matrix")


def fig_power():
    """可检出最小差异随种子数的变化，三条曲线对应三个标准差水平。"""
    fig, ax = fs.new(6.4, 3.7)
    n = np.arange(2, 13)
    label = {0.002: "$\\sigma=.002$", 0.003: "$\\sigma=.003$", 0.005: "$\\sigma=.005$"}
    for s, c in ((0.002, fs.BLUE), (0.003, fs.ACCENT), (0.005, fs.PURPLE)):
        mde = st.t.ppf(0.975, n - 1) * s * np.sqrt(2.0 / n)
        ax.plot(n, mde, marker="o", ms=3.4, lw=1.5, color=c, label=label[s], zorder=3)
    ax.axhline(0.01, color=fs.INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.fill_between([1.7, 12.3], 0.01, 0.075, color=fs.ACCENT, alpha=0.055, zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(1.7, 12.3)
    ax.set_ylim(0.0018, 0.075)
    ax.set_xlabel("每格种子数", fontsize=9.4, labelpad=5)
    ax.set_ylabel("可检出的最小差异（对数刻度）", fontsize=9.4, labelpad=4)
    ax.set_yticks([0.002, 0.005, 0.01, 0.02, 0.05])
    ax.set_yticklabels(["$.002$", "$.005$", "$.01$", "$.02$", "$.05$"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_xticks([2, 3, 4, 6, 8, 10, 12])
    fs.clean(ax, grid=True)
    ax.plot([3], [0.01054], marker="o", ms=7.5, mfc="white", mec=fs.ACCENT, mew=1.6, zorder=5)
    ax.annotate("三个种子 $.0105$", xy=(3, 0.01054), xytext=(3.55, 0.0195),
                fontsize=8.4, color=fs.ACCENT,
                arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.0,
                                shrinkA=1, shrinkB=5, mutation_scale=9))
    fs.label(ax, 11.9, 0.0112, "$\\Delta=.01$ 的门槛", fs=8.4, color=fs.INK,
             ha="right", va="bottom")
    fs.label(ax, 11.9, 0.0355, "这一区检不出", fs=8.4, color=fs.ACCENT, ha="right", va="center")
    ax.legend(loc="lower left", fontsize=8.6, handlelength=1.8, labelspacing=0.42,
              borderaxespad=0.4, frameon=False)
    return fs.save(fig, "theory", "ch10", "power")


def fig_pipeline():
    """两条估计流水线，各自从观测出发，末端给出 delta 与 rho。"""
    fig, ax = fs.canvas(7.4, 3.8)
    lanes = [
        (0.725, "(a) 间隙 $\\delta$", 0.200, fs.LIGHT,
         ["观测对\n计数成 $W_m$", "归并到状态\n得 $\\pi_W$", "对称 Sinkhorn\n行和回到 $p$",
          "谱分解取\n$\\mu_k-\\mu_{k+1}$"]),
        (0.375, "(b) 解释比例 $\\rho$", 0.152, "#FFFFFF",
         ["观测\n嵌入 $F$", "取前 $k$ 主\n方向并白化", "组均值 $m_z$\n组内方差 $\\hat W$",
          "$\\hat C_f$ 减去\n$\\hat W/n_0$", "特征值裁到\n$[0,1]$ 求和"]),
    ]
    gx = 0.025
    for y, tag, bw, fc0, texts in lanes:
        fs.label(ax, 0.012, y + 0.168, tag, fs=9.4, color=fs.MUTED, ha="left", va="bottom")
        xs = []
        for i, t in enumerate(texts):
            x = 0.012 + i * (bw + gx)
            xs.append(x)
            fs.box(ax, x, y, bw, 0.145, t, fc=fc0 if i == 0 else "#FFFFFF",
                   ec="#94A3B8", fs=8.2, lw=0.9)
        for i in range(len(texts) - 1):
            fs.arrow(ax, (xs[i] + bw + 0.003, y + 0.072), (xs[i + 1] - 0.003, y + 0.072),
                     color="#94A3B8", lw=1.0, ms=7)
        ax.text(xs[-1] + bw + 0.016, y + 0.072, "$\\delta$" if y > 0.6 else "$\\rho$",
                fontsize=13, color=fs.ACCENT, ha="left", va="center", zorder=4)
    fs.label(ax, 0.012, 0.978, "取间隙之前先查边缘误差，压到 $10^{-13}$ 以下",
             fs=8.2, color=fs.FAINT, ha="left", va="center")
    fs.label(ax, 0.012, 0.320, "$n_0$ 与分组决定偏差，比较时两组的 $n$ 要一样",
             fs=8.2, color=fs.FAINT, ha="left", va="center")
    fs.box(ax, 0.012, 0.095, 0.876, 0.115,
           "两个量都随 $k$ 变，报告给 $\\delta(k)$ 与 $\\rho(k)$ 两条曲线",
           fc=fs.PAPER, ec=fs.ACCENT, fs=8.8, lw=1.0)
    return fs.save(fig, "theory", "ch10", "pipeline")


def fig_report():
    """结果报告模板的字段示意，右侧三条批注指向最常缺的三行。"""
    fig, ax = fs.canvas(7.4, 4.2)
    ax.add_patch(Rectangle((0.020, 0.110), 0.600, 0.820, facecolor="white",
                           edgecolor="#94A3B8", lw=1.0, zorder=2))
    ax.add_patch(Rectangle((0.020, 0.855), 0.600, 0.075, facecolor=fs.LIGHT,
                           edgecolor="#94A3B8", lw=1.0, zorder=3))
    fs.label(ax, 0.040, 0.892, "单格结果", fs=10.0, color=fs.INK, ha="left", va="center")
    fs.label(ax, 0.600, 0.892, "S2 / 种子 3", fs=8.2, color=fs.MUTED, ha="right", va="center")
    rows = [
        ("设置", "视图、$k$、状态数、$n_0$", False),
        ("主指标", "$.044\\pm.003$（$n=5$ 种子）", True),
        ("两个估计量", "$\\hat\\delta=.52$，$\\hat\\rho=.66$", False),
        ("定理右端", "代入 $\\hat\\delta$ 得 $1.7$", True),
        ("实测误差", "$0.31$", False),
        ("噪声地板", "换种子重跑差 $.0096$", True),
        ("判定", "界有信息量，主对比成立", False),
    ]
    ytop = 0.855
    step = 0.108
    for i, (k, v, flag) in enumerate(rows):
        y = ytop - (i + 1) * step
        if i:
            ax.plot([0.020, 0.620], [y + step, y + step], color=fs.GRID, lw=0.8, zorder=3)
        fs.label(ax, 0.042, y + 0.052, k, fs=8.8, color=fs.INK, ha="left", va="center")
        fs.label(ax, 0.165, y + 0.052, v, fs=8.4, color=fs.MUTED, ha="left", va="center")
        if flag:
            ax.add_patch(Rectangle((0.028, y + 0.020), 0.560, 0.068, facecolor="none",
                                   edgecolor=fs.ACCENT, lw=1.0, ls=(0, (3, 2)), zorder=4))
    notes = [
        (ytop - 2 * step, "不写 $n$，方差无从判断"),
        (ytop - 4 * step, "不写界，不知它有无信息量"),
        (ytop - 6 * step, "不写地板，差异可能只是种子"),
    ]
    for y, t in notes:
        ax.annotate(t, xy=(0.590, y + 0.052), xytext=(0.660, y + 0.052),
                    fontsize=8.2, color=fs.ACCENT, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                    shrinkA=2, shrinkB=2, mutation_scale=8))
    fs.label(ax, 0.020, 0.048, "七行缺一行，读者就无法复核这一格", fs=8.4,
             color=fs.MUTED, ha="left", va="center")
    return fs.save(fig, "theory", "ch10", "report")


def spectral_gap(Pi, p, k):
    """与正文代码骨架一致：对称 Sinkhorn 之后取第 k 与第 k+1 个特征值之差。"""
    a = np.ones_like(p)
    for _ in range(500):
        a = a * np.sqrt(p / (Pi * np.outer(a, a)).sum(1))
    K = Pi * np.outer(a, a) / np.sqrt(np.outer(p, p))
    mu = np.linalg.eigvalsh(K)[::-1][1:]
    return mu[k - 1] - (mu[k] if k < len(mu) else -np.inf)


def scene_fraction(F, gid, k):
    """与正文代码骨架一致：白化、组均值、扣掉组内方差、截断后取迹。"""
    F = F[:, :k] - F[:, :k].mean(0)
    L = np.linalg.cholesky(np.cov(F.T) + 1e-8 * np.eye(k))
    Z = np.linalg.solve(L, F.T).T
    ids = np.unique(gid)
    M = np.stack([Z[gid == g].mean(0) for g in ids])
    W = np.mean([np.cov(Z[gid == g].T, bias=True) for g in ids], 0)
    lam = np.clip(np.linalg.eigvalsh(np.cov(M.T) - W / (len(Z) / len(ids))), 0, 1)
    return np.sqrt(lam).sum() / k


def check_numbers():
    """复算 10.1 与 10.2 引用的两个读数：两簇世界上的 delta，与三个噪声幅度下的 rho。"""
    rng = np.random.default_rng(0)
    n, k = 24, 8
    c = np.repeat([0, 1], n // 2)
    Pi = np.where(c[:, None] == c[None, :], rng.uniform(.6, 1., (n, n)),
                  rng.uniform(.01, .05, (n, n)))
    Pi = (Pi + Pi.T) / 2
    np.fill_diagonal(Pi, 0.)
    p = rng.uniform(.3, 1., n)
    p /= p.sum()
    print("delta(1)=%.3f  delta(2)=%.3f" % (spectral_gap(Pi, p, 1), spectral_gap(Pi, p, 2)))
    g = np.repeat(np.arange(n), 5)
    for noise in (0.3, 0.7, 1.2):
        z = np.repeat(np.eye(n, k), 5, axis=0) @ rng.normal(size=(k, k))
        F = z + noise * rng.normal(size=(len(g), k))
        print("noise %.1f  rho=%.3f" % (noise, scene_fraction(F, g, k)))


if __name__ == "__main__":
    fig_design_matrix()
    fig_power()
    fig_pipeline()
    fig_report()
    check_numbers()
