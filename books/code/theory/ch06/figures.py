# -*- coding: utf-8 -*-
"""第 6 章插图：合成世界上的到达半径、诱导世界核。

图 6-1 是按论文（Figure 4 及其讨论）的结论重绘的示意图，刻度与定性结构与论文一致，
色带由本书按论文表述构造，不是原始实验数据。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Ellipse
import figstyle as fs

# 与论文色阶同向的低饱和色带：浅米（对齐差）→ 深靛（对齐好）
CMAP = LinearSegmentedColormap.from_list(
    "align", ["#FBF7F0", "#EBEFF8", "#BCD0EE", "#6E93D8", "#2C50B4", "#16225C"])


# ---------------------------------------------------------------- 图 6-1
def fig_reach():
    """合成世界上，坏视图对齐度随样本数 N 与到达半径 Δ 的变化。"""
    N = np.logspace(0, np.log10(64), 320)      # 每个状态的样本数
    D = np.logspace(-3, 0, 320)                # 坏视图的到达半径
    NN, DD = np.meshgrid(N, D)

    # 需要的到达半径：一个状态宽 1/6，样本间最长间隙的量级是 log N / N
    need = 0.15 * np.log(np.e * NN) / NN
    ratio = DD / need                          # 实际到达半径相对需求的倍数

    # 跨类之后的软上限（.95），到不了 1
    ceil = 1.0 - 0.05 / (1.0 + np.exp(-(np.log10(DD) + 0.55) / 0.30))
    align = ceil * (1.0 - np.exp(-1.6 * ratio))

    fig = plt.figure(figsize=(6.7, 4.65))
    ax = fig.add_axes([0.118, 0.235, 0.655, 0.615])
    axt = fig.add_axes([0.118, 0.895, 0.655, 0.048])

    # 顶条：好视图，任意样本数都精确对齐
    axt.imshow(np.ones((1, 2)), aspect="auto", cmap=CMAP, vmin=0.0, vmax=1.0)
    axt.set_xticks([]); axt.set_yticks([])
    for s in axt.spines.values():
        s.set_visible(False)
    axt.text(0.5, 0.5, "好视图　任意样本数下都精确对齐（对齐度 1）", transform=axt.transAxes,
             fontsize=8.2, color="white", ha="center", va="center")

    mesh = ax.pcolormesh(NN, DD, align, cmap=CMAP, vmin=0.0, vmax=1.0,
                         shading="gouraud", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 64); ax.set_ylim(1e-3, 1)
    ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(["1", "4", "16", "64"])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_yticklabels([".001", ".01", ".1", "1"])
    ax.set_xlabel("每个状态的样本数 $N$（对数刻度）", labelpad=6)
    ax.set_ylabel("到达半径 $\\Delta$（对数刻度）", labelpad=4)

    # 0.9 等值线（白）与 Δ ∝ log N / N（虚线）
    cs = ax.contour(NN, DD, align, levels=[0.9], colors="white", linewidths=1.35, zorder=4)
    ax.clabel(cs, fmt={0.9: "0.9"}, inline=True, fontsize=8.0,
              manual=[(20.0, 0.040)], colors="white")
    curve = 0.15 * np.log(np.e * N) / N
    ax.plot(N, curve, color=fs.PAPER, lw=1.25, ls=(0, (4.5, 3.2)), zorder=5)

    # 标注放在深色区，读起来不费力
    fs.label(ax, 1.30, 0.475, "需要的到达半径", fs=8.2, color="white", ha="left")
    fs.label(ax, 1.30, 0.345, "$\\Delta\\propto\\log N/N$", fs=9.0, color="white", ha="left")

    # 跨类上限的标注
    fs.label(ax, 8.2, 0.60, "到达半径跨过类间空隙之后", fs=8.2, color="white", ha="left")
    fs.label(ax, 8.2, 0.43, "上限 .95，样本再多也不回到 1", fs=8.2, color="white", ha="left")

    # 右侧三个到达半径的刻度说明
    ticks = [(0.005, "弱 $\\Delta=.005$"), (0.08, "中 $\\Delta=.08$"), (1.0, "强 $\\Delta=1$")]
    for yv, txt in ticks:
        fy = np.log10(yv / 1e-3) / 3.0
        ax.plot([1.008, 1.018], [fy, fy], transform=ax.transAxes, color=fs.FAINT,
                lw=0.9, clip_on=False)
        ax.text(1.030, fy, txt, transform=ax.transAxes, fontsize=7.6, color=fs.MUTED,
                ha="left", va="center", clip_on=False)

    cax = fig.add_axes([0.118, 0.098, 0.30, 0.024])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                      ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label("对齐度（平均典型相关，$k=4$）", labelpad=5, fontsize=8.6)
    cb.ax.tick_params(labelsize=8, length=2.5)
    cb.outline.set_visible(False)
    return fs.save(fig, "theory", "ch06", "reach-heatmap")


# ---------------------------------------------------------------- 图 6-2
def fig_kernel():
    """从观测上的正样本到状态之间的诱导核。"""
    fig = plt.figure(figsize=(6.9, 3.65))

    # ---- 左：观测层，两个类各两个状态，每个状态三个观测
    ax = fig.add_axes([0.010, 0.070, 0.300, 0.820])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs.label(ax, 0.50, 0.972, "观测层", fs=8.8, color=fs.INK, weight="bold")
    fs.label(ax, 0.50, 0.918, "一个状态三个观测，边是正样本", fs=7.6, color=fs.MUTED)

    def block(cy, tag):
        ax.add_patch(Ellipse((0.50, cy), 0.82, 0.30, fill=False, ls=(0, (3, 2.6)),
                             lw=0.9, edgecolor=fs.FAINT, zorder=1))
        fs.label(ax, 0.295, cy + 0.108, tag, fs=8.0, color=fs.MUTED, ha="left")
        return [[(0.245 + k * 0.155, cy + 0.052 - si * 0.108) for k in range(3)]
                for si in range(2)]

    groups = [(block(0.735, "类甲"), 1), (block(0.285, "类乙"), 3)]

    for pts, first in groups:
        for row in pts:
            for i in range(3):
                for j in range(i + 1, 3):
                    ax.plot([row[i][0], row[j][0]], [row[i][1], row[j][1]],
                            color=fs.BLUE, lw=1.15, zorder=2, solid_capstyle="round")
        # 两个状态之间偶发的一条边：靠损坏重叠碰出来
        ax.plot([pts[0][2][0], pts[1][0][0]], [pts[0][2][1], pts[1][0][1]],
                color=fs.BLUE, lw=1.15, zorder=2, solid_capstyle="round")
        for si, row in enumerate(pts):
            fs.label(ax, 0.665, pts[si][0][1], "状态 %d" % (first + si), fs=7.2,
                     color=fs.MUTED, ha="left")
            for p in row:
                ax.scatter([p[0]], [p[1]], s=25, color=fs.INK, zorder=4,
                           edgecolor="white", linewidth=0.7)

    fs.label(ax, 0.50, 0.045, "正样本只在类内把观测连起来", fs=7.6, color=fs.BLUE)

    # ---- 中间：按状态归并
    ar = FancyArrowPatch((0.318, 0.520), (0.408, 0.520), transform=fig.transFigure,
                         arrowstyle="-|>", mutation_scale=8.5, lw=1.0, color=fs.FAINT,
                         shrinkA=0, shrinkB=0)
    fig.add_artist(ar)
    fig.text(0.380, 0.575, "按状态归并", fontsize=7.6, color=fs.MUTED, ha="center")

    # ---- 右：四个 4×4 诱导核
    mats = [
        ("(a)", "好视图", "$\\tau=0$，回到世界核",
         [[.50, .50, .00, .00], [.50, .50, .00, .00],
          [.00, .00, .50, .50], [.00, .00, .50, .50]]),
        ("(b)", "弱损坏，样本少", "核近乎对角",
         [[.88, .12, .00, .00], [.12, .88, .00, .00],
          [.00, .00, .88, .12], [.00, .00, .12, .88]]),
        ("(c)", "弱损坏，样本多", "类内连通",
         [[.62, .38, .00, .00], [.38, .62, .00, .00],
          [.00, .00, .62, .38], [.00, .00, .38, .62]]),
        ("(d)", "强损坏", "跨类也有质量",
         [[.31, .25, .22, .22], [.25, .31, .22, .22],
          [.22, .22, .31, .25], [.22, .22, .25, .31]]),
    ]

    w, gap, h, ybot = 0.112, 0.0145, 0.212, 0.455
    x0 = 0.452
    for i, (tag, name, note, M) in enumerate(mats):
        x = x0 + i * (w + gap)
        axm = fig.add_axes([x, ybot, w, h])
        axm.imshow(np.array(M), cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")
        for k in range(5):
            axm.axhline(k - 0.5, color="white", lw=0.7)
            axm.axvline(k - 0.5, color="white", lw=0.7)
        for r in range(4):
            for c in range(4):
                v = M[r][c]
                axm.text(c, r, ".%02d" % round(v * 100), ha="center", va="center",
                         fontsize=6.5, color="white" if v > .42 else fs.INK)
        axm.set_xticks([]); axm.set_yticks([])
        for s in axm.spines.values():
            s.set_edgecolor(fs.FAINT); s.set_linewidth(0.8)
        if i == 0:
            axm.set_xticks([0, 1, 2, 3]); axm.set_xticklabels(["1", "2", "3", "4"], fontsize=7.0)
            axm.set_yticks([0, 1, 2, 3]); axm.set_yticklabels(["1", "2", "3", "4"], fontsize=7.0)
            axm.tick_params(length=0, pad=1.5)
        axm.text(0.5, 1.10, tag, transform=axm.transAxes, fontsize=8.0, color=fs.INK,
                 weight="bold", ha="center", va="center")
        fig.text(x + w / 2, ybot - 0.070, name, fontsize=7.4, color=fs.INK, ha="center")
        fig.text(x + w / 2, ybot - 0.128, note, fontsize=7.0, color=fs.MUTED, ha="center")

    # 类块标记
    axb = fig.add_axes([x0, ybot + h + 0.078, 4 * w + 3 * gap, 0.050])
    axb.set_xlim(0, 4); axb.set_ylim(0, 1); axb.axis("off")
    for a, b, tag in ((0.06, 1.94, "类甲两个状态"), (2.06, 3.94, "类乙两个状态")):
        axb.plot([a, b], [0.30, 0.30], color=fs.MUTED, lw=0.8)
        axb.plot([a, a], [0.30, 0.14], color=fs.MUTED, lw=0.8)
        axb.plot([b, b], [0.30, 0.14], color=fs.MUTED, lw=0.8)
        axb.text((a + b) / 2, 0.60, tag, fontsize=7.4, color=fs.MUTED, ha="center")

    fig.text(0.452, 0.930, "行与列都是状态，格子是 $\\pi_{W_m}$ 的行归一化值",
             fontsize=8.4, color=fs.INK, ha="left")
    fig.text(0.452, 0.245, "颜色越深表示这一对状态被一个正样本连上的概率越高",
             fontsize=7.4, color=fs.MUTED, ha="left")
    return fs.save(fig, "theory", "ch06", "induced-kernel")


# ---------------------------------------------------------------- 图 6-3
def fig_law_residual():
    """缩放律的拟合残差。数字全部取自论文正文与附录，逐点的 R@1 没有随文发布。"""
    fig = plt.figure(figsize=(6.9, 3.45))

    # ---- 左：四组均方根误差
    axl = fig.add_axes([0.283, 0.185, 0.318, 0.735])
    rows = [
        ("拟合的 45 个格子", .007, fs.BLUE),
        ("留出的 8 个格子", .0115, fs.FAINT),
        ("预测的 85 次运行", .0093, fs.BLUE),
        ("其中 $q\\geq .1$", .0065, "#7FA6E8"),
        ("其中 $.01\\leq q<.1$", .0071, "#7FA6E8"),
        ("其中 $q<.01$", .012, fs.ACCENT),
    ]
    ys = np.arange(len(rows))
    for y, (name, val, col) in zip(ys, rows):
        axl.barh(y, val, height=.56, color=col, edgecolor="none", zorder=3)
        lab = ("%.4f" % val)[1:].rstrip("0")
        axl.text(val + .0006, y, lab, va="center", ha="left", fontsize=8.0,
                 color=fs.INK, zorder=4)
    axl.set_yticks(ys)
    axl.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    axl.set_ylim(-.7, len(rows) - .3)
    axl.set_xlim(0, .0172)
    axl.set_xticks([0, .004, .008, .012, .016])
    axl.set_xticklabels(["0", ".004", ".008", ".012", ".016"], fontsize=8.0)
    axl.set_xlabel("均方根误差（R@1）", fontsize=8.6, labelpad=5)
    axl.axvline(.0096, color=fs.MUTED, lw=1.0, ls=(0, (4, 2.6)), zorder=2)
    fs.clean(axl, grid=False)
    axl.tick_params(length=2.5)
    axl.text(.0104, -0.52, ".0096", fontsize=7.8, color=fs.MUTED, ha="left", va="center")
    axl.text(0.0, 1.045, "(a)", transform=axl.transAxes, fontsize=9.0, color=fs.INK,
             weight="bold", ha="left", va="bottom")

    # ---- 右：3.2 亿个样本处的实测减预测
    axr = fig.add_axes([0.700, 0.185, 0.278, 0.735])
    pts = [("0", 0.001), (".1", -0.041), ("1", -0.037)]
    xs = np.arange(len(pts))
    for x, (name, res) in zip(xs, pts):
        col = fs.ACCENT if res < -.01 else fs.GREEN
        axr.bar(x, res, width=.46, color=col, edgecolor="none", zorder=3)
        dy = -.004 if res < 0 else .004
        lab = ("\u2212" + ("%.3f" % -res)[1:]) if res < 0 else ("+" + ("%.3f" % res)[1:])
        axr.text(x, res + dy, lab, ha="center", va="top" if res < 0 else "bottom",
                 fontsize=8.2, color=fs.INK, zorder=4)
    axr.axhline(0, color=fs.FAINT, lw=0.9, zorder=2)
    axr.set_xticks(xs)
    axr.set_xticklabels(["$q=%s$" % p[0] for p in pts], fontsize=8.4)
    axr.set_xlim(-.62, 2.62)
    axr.set_ylim(-.052, .020)
    axr.set_yticks([-.04, -.02, 0, .02])
    axr.set_yticklabels(["−.04", "−.02", "0", ".02"], fontsize=8.0)
    axr.set_xlabel("好视图比例", fontsize=8.6, labelpad=5)
    axr.set_ylabel("实测减预测（R@1）", fontsize=8.6, labelpad=6)
    fs.clean(axr, grid=False)
    axr.tick_params(length=2.5)
    fs.label(axr, 1.35, .0145, "3.2 亿个样本，拟合范围的四倍", fs=7.8, color=fs.MUTED)
    axr.text(0.0, 1.045, "(b)", transform=axr.transAxes, fontsize=9.0, color=fs.INK,
             weight="bold", ha="left", va="bottom")
    return fs.save(fig, "theory", "ch06", "law-residual")


# ---------------------------------------------------------------- 图 6-4
def fig_tau_epsilon():
    """定理 4 的两项：τ 随混合程度与间隙的变化，以及两条范数支的取小。"""
    k = 4
    fig = plt.figure(figsize=(6.9, 3.30))

    # ---- 左：τ 随 ε 的斜率是 2√2√k / δ
    axl = fig.add_axes([0.085, 0.180, 0.395, 0.720])
    eps = np.linspace(0, 0.30, 400)
    trivial = np.sqrt(2 * k)
    for delta, col in [(.5, fs.BLUE), (.2, fs.TEAL), (.1, fs.GOLD), (.05, fs.ACCENT)]:
        tau = 2 * np.sqrt(2) * np.sqrt(k) * eps / delta
        axl.plot(eps, tau, color=col, lw=1.5, zorder=3, label="$\\delta=%.2g$" % delta)
    axl.axhspan(trivial, 3.55, color=fs.LIGHT, zorder=1)
    axl.axhline(trivial, color=fs.MUTED, lw=1.0, ls=(0, (4, 2.6)), zorder=2)
    axl.set_xlim(0, .30)
    axl.set_ylim(0, 3.55)
    axl.set_xticks([0, .1, .2, .3])
    axl.set_xticklabels(["0", ".1", ".2", ".3"], fontsize=8.0)
    axl.set_yticks([0, 1, 2, 3])
    axl.set_xlabel("混合程度 $\\varepsilon_m$", fontsize=8.8, labelpad=5)
    axl.set_ylabel("$\\tau_m$", fontsize=9.2, labelpad=4)
    fs.clean(axl, grid=False)
    axl.tick_params(length=2.5)
    fs.label(axl, .152, 3.36, "界在这里没有信息量", fs=7.8, color=fs.MUTED)
    fs.label(axl, .205, 2.60, "$\\sqrt{2k}=2.83$", fs=8.0, color=fs.MUTED, ha="right")
    axl.legend(loc="lower right", fontsize=8.0, handlelength=1.7, borderpad=0.3,
               labelspacing=0.35, bbox_to_anchor=(0.985, 0.035))
    axl.text(0.0, 1.055, "(a)", transform=axl.transAxes, fontsize=9.0, color=fs.INK,
             weight="bold", ha="left", va="bottom")

    # ---- 右：两条范数支的相对大小，横轴是 ε^HS / ε
    axr = fig.add_axes([0.590, 0.180, 0.388, 0.720])
    rho = np.linspace(1.0, 3.0, 400)
    op = np.ones_like(rho)                      # √k ε，归一化后恒为 1
    hs = rho / np.sqrt(k / 2.0)                 # √2 ε^HS，同样归一化
    axr.plot(rho, op, color=fs.BLUE, lw=1.4, ls=(0, (5, 2.4)), zorder=3,
             label="$\\sqrt{k}\\varepsilon_m$（算子范数）")
    axr.plot(rho, hs, color=fs.GOLD, lw=1.4, ls=(0, (1.2, 2.0)), zorder=3,
             label="$\\sqrt{2}\\varepsilon_m^{\\mathrm{HS}}$（HS 范数）")
    axr.plot(rho, np.minimum(op, hs), color=fs.INK, lw=2.0, zorder=4, label="两者取小")
    cross = np.sqrt(k / 2.0)
    axr.plot([cross, cross], [0, 1.0], color=fs.MUTED, lw=0.9, ls=(0, (3, 2.5)), zorder=2)
    axr.scatter([cross], [1.0], s=26, color=fs.INK, zorder=5, edgecolor="white", linewidth=0.8)
    axr.set_xlim(1.0, 3.0)
    axr.set_ylim(0, 2.15)
    axr.set_xticks([1, 1.5, 2, 2.5, 3])
    axr.set_xticklabels(["1", "1.5", "2", "2.5", "3"], fontsize=8.0)
    axr.set_yticks([0, .5, 1, 1.5, 2])
    axr.set_yticklabels(["0", "0.5", "1", "1.5", "2"], fontsize=8.0)
    axr.set_xlabel("$\\varepsilon_m^{\\mathrm{HS}}/\\varepsilon_m$", fontsize=9.0, labelpad=5)
    axr.set_ylabel("混合项（以 $\\sqrt{k}\\,\\varepsilon_m$ 为单位）", fontsize=8.8, labelpad=6)
    fs.clean(axr, grid=False)
    axr.tick_params(length=2.5)
    axr.legend(loc="lower right", fontsize=7.6, handlelength=1.8, borderpad=0.3,
               labelspacing=0.35, bbox_to_anchor=(0.985, 0.035))
    axr.annotate("折点在 $\\sqrt{k/2}=1.41$", xy=(cross, 0.97), xytext=(1.03, 0.30),
                 fontsize=8.0, color=fs.INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=fs.FAINT, lw=0.9,
                                 shrinkA=2, shrinkB=3, mutation_scale=7))
    axr.text(0.0, 1.055, "(b)", transform=axr.transAxes, fontsize=9.0, color=fs.INK,
             weight="bold", ha="left", va="bottom")
    return fs.save(fig, "theory", "ch06", "tau-epsilon")


if __name__ == "__main__":
    fig_reach()
    fig_kernel()
    fig_law_residual()
    fig_tau_epsilon()
