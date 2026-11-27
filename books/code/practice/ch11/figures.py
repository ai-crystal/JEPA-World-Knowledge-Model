# -*- coding: utf-8 -*-
"""第 11 章插图：塌缩诊断树、超参敏感性网格、消融森林图、两次训练曲线对照。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import figstyle as fs

# 温色阶：米色（差）→ 陶土（好），与 fs.ACCENT 同向但压低饱和
CMAP = LinearSegmentedColormap.from_list(
    "sweep", ["#FCF9F3", "#F6E9D8", "#EBD2B4", "#D9AE85", "#BE7F52", "#8E4A24"])


# ---------------------------------------------------------------- 图 11-1
# 三列症状，每列三步检查 + 一步处置
TREE = [
    ("损失收敛到底　探针≈随机", fs.ACCENT, "完全塌缩或结构缺失",
     [("①", "第 20 步打印梯度范数\n查停梯度、目标网络"),
      ("②", "通道标准差是否低于\n1/√d 的一半"),
      ("③", "谱线看尾部能量与最大\n奇异值，别只看有效秩")],
     "补结构约束：方差项、\n白化或停梯度；\n增强先降一档"),
    ("损失不降或振荡　出现 NaN", fs.BLUE, "优化与数值问题",
     [("①", "单批过拟合：lr 置零\n固定一批应降到零"),
      ("②", "lr 除 3 重跑\n看损失是否单调"),
      ("③", "看裁剪触发率\n与第一个 NaN 的层")],
     "降 lr、加预热、开裁剪\n仍炸则查数据管道\n与混合精度"),
    ("损失正常　探针卡在半高", fs.GREEN, "表征上限，不是故障",
     [("①", "先别动优化器\n去查视图的裁剪强度"),
      ("②", "有效秩与 ρ 是否\n低于同规模基线"),
      ("③", "种子间方差是否\n大于要检出的差值")],
     "换更强的好视图\n或承认已到容量上限\n去加数据不加宽"),
]


def fig_triage():
    fig, ax = fs.canvas(7.2, 4.35)

    fs.box(ax, 0.320, 0.900, 0.360, 0.072, "冻结编码器，三个读数一起看",
           fc=fs.LIGHT, ec=fs.MUTED, fs=9.2, rounded=0.014)

    cxs = [0.172, 0.500, 0.828]
    bw = 0.292
    for (head, color, sub, checks, action), cx in zip(TREE, cxs):
        x = cx - bw / 2
        # 症状
        fs.box(ax, x, 0.788, bw, 0.083, head, fc="white", ec=color, fs=8.4,
               weight="bold", rounded=0.012)
        fs.arrow(ax, (cx, 0.899), (cx, 0.874), color=fs.FAINT, lw=1.0, ms=7)

        # 三步检查
        ys = [0.662, 0.536, 0.410]
        for (num, txt), y in zip(checks, ys):
            fs.box(ax, x, y, bw, 0.108, "", fc="#FFFFFF", ec="#D1D5DB", lw=0.8,
                   rounded=0.010)
            fs.label(ax, x + 0.016, y + 0.054, num, fs=8.2, color=color,
                     weight="bold", ha="left")
            fs.label(ax, x + 0.042, y + 0.054, txt, fs=7.3, color=fs.INK,
                     ha="left", va="center")
        for y0, y1 in ((0.786, 0.772), (0.660, 0.646), (0.534, 0.520)):
            fs.arrow(ax, (cx, y0), (cx, y1), color=fs.FAINT, lw=0.9, ms=6)

        # 处置
        fs.box(ax, x, 0.108, bw, 0.212, "", fc=color, ec=color, rounded=0.014, alpha=0.93)
        fs.label(ax, cx, 0.270, sub, fs=7.6, color="white", weight="bold", z=6)
        fs.label(ax, cx, 0.188, action, fs=7.2, color="white", z=6)
        fs.arrow(ax, (cx, 0.408), (cx, 0.324), color=color, lw=1.2, ms=8)

    fs.label(ax, 0.008, 0.055, "三格同时恶化才判塌缩；只有一格异常时，先怀疑测量口径（批大小、验证集、钩子）。",
             fs=7.4, color=fs.MUTED, ha="left")
    return fs.save(fig, "practice", "ch11", "triage")


# ---------------------------------------------------------------- 图 11-2
LRS = [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3]
WDS = [0.01, 0.03, 0.10, 0.30, 1.00]

GRID_STRONG = np.array([   # λ = 0.05
    [10.6, 14.8, 18.9, 20.4, 17.2],
    [13.1, 18.4, 22.6, 23.1, 18.9],
    [15.2, 21.3, 24.8, 24.1, 19.6],
    [14.0, 20.4, 23.9, 21.2, 16.4],
    [10.2, 16.1, 18.4, 14.3, 10.1],
])

GRID_WEAK = np.array([     # λ = 0.005
    [1.6, 2.9, 11.4, 17.6, 19.8],
    [1.5, 5.8, 17.2, 21.0, 20.4],
    [2.4, 11.6, 21.5, 22.6, 18.1],
    [8.1, 17.9, 22.1, 19.8, 12.6],
    [14.9, 19.4, 17.2, 11.4, 6.8],
])


def _panel(ax, grid, title):
    # grid 的行是学习率、列是权重衰减；画在坐标轴上要转置成 x=lr, y=wd
    G = grid.T
    im = ax.imshow(G, origin="lower", cmap=CMAP, vmin=0, vmax=26,
                   extent=(-0.5, 4.5, -0.5, 4.5), aspect="auto", zorder=1)
    for i in range(5):
        for j in range(5):
            v = G[i, j]
            lum = im.cmap(im.norm(v))[:3]
            dark = 0.299 * lum[0] + 0.587 * lum[1] + 0.114 * lum[2] < 0.55
            ax.text(j + 0.42, i + 0.40, "%.1f" % v, ha="right", va="top", fontsize=7.4,
                    color="white" if dark else fs.INK, zorder=4)
    # 每一列（固定学习率）里最好的权重衰减，用白色虚线连成脊
    best = [int(np.argmax(G[:, j])) for j in range(5)]
    ax.plot(range(5), best, color="white", lw=1.3, ls=(0, (2.5, 2.0)), zorder=5)
    for j, i in enumerate(best):
        ax.plot([j], [i], marker="o", ms=4.6, mfc="none", mec="white", mew=1.4, zorder=6)

    ax.set_xticks(range(5))
    ax.set_xticklabels(["2.5e−4", "5e−4", "1e−3", "2e−3", "4e−3"], fontsize=7.4)
    ax.set_yticks(range(5))
    ax.set_yticklabels([".01", ".03", ".10", ".30", "1.0"], fontsize=7.4)
    ax.set_xlabel("学习率", fontsize=8.2, labelpad=3)
    ax.set_title(title, fontsize=9.0, color=fs.INK, pad=6)
    ax.tick_params(length=2, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_sensitivity():
    fig = plt.figure(figsize=(7.2, 3.45))
    ax1 = fig.add_axes([0.070, 0.215, 0.352, 0.660])
    ax2 = fig.add_axes([0.578, 0.215, 0.352, 0.660])
    _panel(ax1, GRID_STRONG, "（a）λ = 0.02")
    _panel(ax2, GRID_WEAK, "（b）λ = 0.005")
    ax1.set_ylabel("权重衰减", fontsize=8.2, labelpad=3)

    fig.text(0.070, 0.088, "格子内是线性探针 top-1（%），机会水平 1%", fontsize=7.2,
             color=fs.MUTED, ha="left")
    fig.text(0.070, 0.030, "网格数值由脚本按示意模型生成，用来看脊的形状，不可照抄", fontsize=7.2,
             color=fs.MUTED, ha="left")

    ax2.add_patch(plt.Rectangle((-0.5, -0.5), 2.0, 2.0, fill=False, lw=1.4,
                                edgecolor=fs.ACCENT, ls=(0, (3, 2)), zorder=7))
    fig.text(0.578, 0.088, "虚线框内四格已与机会水平无异", fontsize=7.2,
             color=fs.ACCENT, ha="left", weight="bold")
    return fs.save(fig, "practice", "ch11", "sensitivity")


# ---------------------------------------------------------------- 图 11-3
ABL = [
    ("增强回到强档（裁剪下限 0.2）", -4.1, -4.8, -3.4, False),
    ("去掉评测用的 EMA 权重", -1.3, -2.0, -0.6, False),
    ("批 512→128，lr 按 √B 重标", -1.2, -1.9, -0.5, False),
    ("权重衰减 0.08→0.03", -0.8, -1.4, -0.2, False),
    ("去掉梯度裁剪 1.0", -0.4, -1.1, 0.3, True),
    ("SIGReg 方向数 1024→256", -0.1, -0.7, 0.5, True),
]


def fig_ablation():
    fig = plt.figure(figsize=(6.9, 3.05))
    ax = fig.add_axes([0.392, 0.255, 0.318, 0.680])
    n = len(ABL)
    for k, (name, d, lo, hi, crosses) in enumerate(ABL):
        y = n - 1 - k
        col = fs.MUTED if crosses else fs.ACCENT
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=3)
        for xx in (lo, hi):
            ax.plot([xx, xx], [y - 0.11, y + 0.11], color=col, lw=1.2, zorder=3)
        ax.plot([d], [y], marker="o", ms=6.4, mfc=col, mec="white", mew=1.1, zorder=5)
        ax.plot([d], [y], marker="o", ms=2.2, mfc="white", mec="none", zorder=6)
        tr = ax.get_yaxis_transform()
        ax.text(-0.03, y, name, transform=tr, ha="right", va="center", fontsize=7.9,
                color=fs.INK if not crosses else fs.MUTED)
        txt = "%+.1f  [%+.1f, %+.1f]" % (d, lo, hi)
        ax.text(1.04, y, txt.replace("-", "−"), transform=tr,
                ha="left", va="center", fontsize=7.4, color=col)
        if name.startswith("去掉梯度裁剪"):
            ax.text(1.04, y - 0.34, "最差种子 −3.8", transform=tr, ha="left", va="center",
                    fontsize=7.0, color=fs.MUTED)

    ax.axvline(0, color=fs.FAINT, lw=1.0, zorder=1)
    ax.set_xlim(-5.6, 1.2)
    ax.set_ylim(-0.75, n - 0.25)
    ax.set_yticks([])
    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1])
    ax.set_xlabel("相对 .250 的差值（个百分点），95% 区间", fontsize=8.2, labelpad=4)
    ax.tick_params(labelsize=7.6, length=2.5, pad=2)
    fs.clean(ax, grid=False, spines=("bottom",))
    ax.spines["bottom"].set_color(fs.FAINT)

    fig.text(0.025, 0.935, "四个种子，同一个起点，只改一行", fontsize=7.4, color=fs.MUTED)
    return fs.save(fig, "practice", "ch11", "ablation")


# ---------------------------------------------------------------- 图 11-4
def _smooth(x, y, n=400):
    xs = np.linspace(x[0], x[-1], n)
    return xs, np.interp(xs, x, y)


def fig_curves():
    ep = np.array([0, 15, 30, 50, 70, 90, 120, 160, 200, 250, 300])

    base = np.array([1.0, 2.4, 5.1, 8.0, 10.4, 11.8, 12.5, 12.8, 12.9, 13.0, 13.1])
    final = np.array([1.0, 3.6, 7.4, 11.9, 15.6, 18.4, 21.0, 23.0, 24.2, 24.8, 25.0])

    sig_base = np.array([0.050, 0.050, 0.049, 0.048, 0.047, 0.046, 0.045, 0.044,
                         0.044, 0.043, 0.043])
    sig_final = np.array([0.050, 0.050, 0.049, 0.049, 0.048, 0.048, 0.047, 0.047,
                          0.047, 0.047, 0.047])

    fig = plt.figure(figsize=(7.1, 3.05))

    # (a) 探针
    ax1 = fig.add_axes([0.082, 0.215, 0.385, 0.665])
    xs, ys = _smooth(ep, base)
    ax1.fill_between(xs, ys - 1.0, ys + 1.2, color=fs.FAINT, alpha=0.30, lw=0, zorder=1)
    ax1.plot(xs, ys, color=fs.MUTED, lw=1.8, zorder=3, label="基线（4 个种子）")
    xs2, ys2 = _smooth(ep, final)
    ax1.fill_between(xs2, ys2 - 0.7, ys2 + 0.6, color=fs.ACCENT, alpha=0.20, lw=0, zorder=1)
    ax1.plot(xs2, ys2, color=fs.ACCENT, lw=1.9, zorder=4, label="终版（4 个种子）")
    ax1.axhline(1.0, color=fs.FAINT, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax1.text(52, 1.6, "机会水平 1.0", ha="left", va="bottom", fontsize=7.0, color=fs.MUTED)
    ax1.set_xlim(0, 300); ax1.set_ylim(0, 29)
    ax1.set_xticks([0, 100, 200, 300])
    ax1.set_yticks([0, 10, 20, 30])
    ax1.set_xlabel("预训练轮次", fontsize=8.2, labelpad=3)
    ax1.set_ylabel("线性探针 top-1（%）", fontsize=8.2, labelpad=3)
    ax1.tick_params(labelsize=7.4, length=2.5, pad=2)
    fs.clean(ax1, grid=True)
    ax1.legend(loc="lower right", fontsize=7.2, handlelength=1.6, borderpad=0.2)
    ax1.set_title("（a）下游读数", fontsize=9.0, pad=6)

    # (b) 通道标准差
    ax2 = fig.add_axes([0.588, 0.215, 0.385, 0.665])
    xs, ys = _smooth(ep, sig_base)
    ax2.plot(xs, ys, color=fs.MUTED, lw=1.8, zorder=3, label="基线均值")
    # 一个种子在第 58 轮跌破
    tt = np.array([0, 20, 40, 56, 60, 66, 80, 120, 200, 300])
    ss = np.array([0.050, 0.049, 0.047, 0.044, 0.021, 0.008, 0.004, 0.0035, 0.0033, 0.0032])
    ax2.plot(tt, ss, color=fs.PINK, lw=1.5, ls=(0, (4, 2)), zorder=4,
             label="改动三的一个种子")
    xs2, ys2 = _smooth(ep, sig_final)
    ax2.plot(xs2, ys2, color=fs.ACCENT, lw=1.9, zorder=5, label="终版（4 个种子）")
    ax2.axhline(1 / np.sqrt(384), color=fs.FAINT, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax2.text(150, 0.0562, "1/√384 ≈ .0510", ha="center", va="bottom",
             fontsize=7.0, color=fs.MUTED)
    ax2.annotate("第 58 轮跌破", xy=(60, 0.0225), xytext=(112, 0.0335),
                 fontsize=7.0, color=fs.PINK,
                 arrowprops=dict(arrowstyle="-|>", color=fs.PINK, lw=0.9,
                                 shrinkA=2, shrinkB=3, mutation_scale=7))
    ax2.set_yscale("log")
    ax2.set_xlim(0, 300); ax2.set_ylim(0.0025, 0.088)
    ax2.set_xticks([0, 100, 200, 300])
    ax2.minorticks_off()
    ax2.set_yticks([0.003, 0.005, 0.01, 0.02, 0.05])
    ax2.set_yticklabels([".003", ".005", ".010", ".020", ".050"])
    ax2.set_xlabel("预训练轮次", fontsize=8.2, labelpad=3)
    ax2.set_ylabel("每通道标准差（对数）", fontsize=8.2, labelpad=3)
    ax2.tick_params(labelsize=7.4, length=2.5, pad=2)
    fs.clean(ax2, grid=True)
    ax2.legend(loc="center right", fontsize=7.2, handlelength=1.6, borderpad=0.25,
               frameon=True, framealpha=0.90, edgecolor="none", borderaxespad=0.3)
    ax2.set_title("（b）塌缩的前兆", fontsize=9.0, pad=6)
    return fs.save(fig, "practice", "ch11", "curves")


if __name__ == "__main__":
    fig_triage()
    fig_sensitivity()
    fig_ablation()
    fig_curves()
