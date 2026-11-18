# -*- coding: utf-8 -*-
"""第 9 章插图：Flickr 三组文本编码器、WIT 规模曲线、42 倍样本效率。

所有数值取自 Balestriero & LeCun, "Proving and Improving the Platonic
Representation Hypothesis with JEPAs"（arXiv:2610.39302）的 Table 1、
Table 2、Table 3 与 Figure 1、Figure 9 及其正文，不做任何插值或补点。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_flickr_bars():
    """图 9-1：Flickr30k 上图像 JEPA 对四种文本编码器的三条指标（Table 1）。"""
    metrics = ["互 10-NN", "R@1", "R@5"]
    chance = [.010, .001, .005]
    series = [
        ("好视图 JEPA", [.062, .044, .133], fs.ACCENT),
        ("坏视图 JEPA", [.047, .027, .095], fs.BLUE),
        ("下一词预测", [.042, .021, .077], fs.GREEN),
        ("下一词·成对描述", [.043, .023, .083], fs.PINK),
        ("随机初始化", [.010, .001, .011], "#C3CAD3"),
    ]
    n = len(series)
    bw = 0.152
    x0 = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for i, (name, vals, col) in enumerate(series):
        xs = x0 + (i - (n - 1) / 2) * (bw + 0.014)
        ax.bar(xs, vals, width=bw, color=col, label=name, zorder=3,
               edgecolor="white", linewidth=0.6)
        for x, v in zip(xs, vals):
            ax.text(x, max(v + 0.004, 0.007), f"{v:.3f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=7.6, color=fs.MUTED, zorder=4)

    ax.set_xticks(x0)
    ax.set_xticklabels([f"{m}\n随机基线 {c:.3f}".replace("0.", ".") for m, c in zip(metrics, chance)],
                       fontsize=9.4, color=fs.INK)
    ax.set_ylim(0, 0.158)
    ax.set_yticks([0, .04, .08, .12])
    ax.set_ylabel("指标数值（越高越好）", fontsize=9.4)
    fs.clean(ax, grid=True)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=8.8,
              handlelength=1.0, columnspacing=1.1, labelspacing=0.35, borderaxespad=0.2)
    return fs.save(fig, "practice", "ch09", "flickr-bars")


def fig_wit_scaling():
    """图 9-2：WIT-1024 上的互 10-NN 随训练图片数上升（Figure 9 与正文）。"""
    m_good = [2.16, 10, 50, 100]
    v_good = [.108, .111, .125, .125]
    m_bad = [2.16, 50]
    v_bad = [.053, .063]
    m_ntp = [2.16, 50]
    v_ntp = [.071, .079]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.set_xscale("log")
    ax.set_xlim(0.016, 210)
    ax.set_ylim(0.022, 0.145)

    # 语言模型参考线（同一评测协议下的对照，不属于本图的数据点）
    for y, col, lab, dy, va in ((.125, fs.PINK, "Qwen2.5-3B　.125", .0035, "bottom"),
                                (.118, fs.GOLD, "Qwen2.5-7B　.118", -.0035, "top")):
        ax.plot([0.016, 210], [y, y], color=col, lw=1.0, ls=(0, (6, 3)), zorder=1)
        ax.text(0.019, y + dy, lab, ha="left", va=va, fontsize=8.4, color=col)

    ax.plot(m_bad, v_bad, color=fs.BLUE, lw=1.4, ls=(0, (4, 2)), marker="s", ms=4.6,
            zorder=4, markerfacecolor="white", markeredgewidth=1.1, label="坏视图 JEPA")
    ax.plot(m_ntp, v_ntp, color=fs.GREEN, lw=1.4, ls=(0, (2, 2)), marker="^", ms=5.2,
            zorder=4, markerfacecolor="white", markeredgewidth=1.1, label="下一词预测")
    ax.plot(m_good, v_good, color=fs.ACCENT, lw=1.9, marker="o", ms=6.2, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0, label="好视图 JEPA，33M")

    # 逐点标数值：落在参考线上的两个点标在上方，其余标在下方
    for x, v, dy, va in ((2.16, .108, -.0042, "top"), (10, .111, -.0042, "top"),
                         (50, .125, .0045, "bottom"), (100, .125, .0045, "bottom")):
        ax.text(x, v + dy, f"{v:.3f}".lstrip("0"), ha="center", va=va,
                fontsize=8.3, color=fs.MUTED, zorder=6)
    for xs, vs in ((m_bad, v_bad), (m_ntp, v_ntp)):
        for x, v in zip(xs, vs):
            ax.text(x, v - .0042, f"{v:.3f}".lstrip("0"), ha="center", va="top",
                    fontsize=8.3, color=fs.MUTED, zorder=6)

    # Flickr 字幕那一点换过语料，不与网页文本连成一条线
    ax.plot([0.029], [.034], marker="D", ms=5.6, color=fs.ACCENT, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0, label="好视图 JEPA，Flickr 字幕")
    ax.text(0.029, .0375, ".034", ha="center", va="bottom", fontsize=8.3, color=fs.MUTED)

    ax.set_xticks([0.03, 2.16, 10, 50, 100])
    ax.set_xticklabels(["0.03\nFlickr", "2.16", "10", "50", "100"])
    ax.set_xlabel("训练用到的图片数（百万，对数刻度）", fontsize=9.4, labelpad=7)
    ax.set_ylabel("与 DINOv2 ViT-B 的互 10-NN", fontsize=9.4)
    ax.set_yticks([.04, .06, .08, .10, .12, .14])
    fs.clean(ax, grid=True)
    ax.tick_params(axis="x", pad=4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8.6,
              handlelength=1.9, columnspacing=2.4, labelspacing=0.5, borderaxespad=0.0)
    return fs.save(fig, "practice", "ch09", "wit-scaling")


def fig_sample_efficiency():
    """图 9-3：同一个 109M 编码器达到同一对齐水平所需的样本数（4.2 节）。"""
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.set_xscale("log")
    ax.set_xlim(9, 2400)
    ax.set_ylim(0, 0.138)

    pts = [(21.6, .109, fs.ACCENT, "好视图 JEPA"), (900, .111, fs.GREEN, "下一词预测")]

    for m, v, col, name in pts:
        ax.plot([m, m], [0, v], color=col, lw=0.9, ls=(0, (2, 3)), zorder=2)
        ax.plot([m], [v], marker="o", ms=10, color=col, zorder=5,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(m, v + .006, f"{v:.3f}".replace("0.", "."), ha="center", va="bottom",
                fontsize=11.5, color=col, weight="bold")
        ax.text(m, v - .010, name, ha="center", va="top", fontsize=9.2, color=col,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.6), zorder=6)

    ax.plot([21.6, 900], [.109, .111], color=fs.FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.text(139, .1145, "同一对齐水平", ha="center", va="bottom", fontsize=9, color=fs.MUTED)

    ax.annotate("", xy=(21.6, .034), xytext=(900, .034),
                arrowprops=dict(arrowstyle="|-|", color=fs.INK, lw=1.0, mutation_scale=8))
    ax.text(139, .043, "样本少 42 倍", ha="center", va="bottom", fontsize=10.5, color=fs.INK)
    ax.text(139, .027, "词元少 14 倍，唯一图片少 435 倍", ha="center", va="top",
            fontsize=8.8, color=fs.MUTED)

    ax.set_xticks([10, 21.6, 100, 900, 2400])
    ax.set_xticklabels(["10", "21.6", "100", "900", "2400"])
    ax.set_xlabel("训练样本数（百万，对数刻度）", fontsize=9.4, labelpad=7)
    ax.set_ylabel("WIT-1024 互 10-NN", fontsize=9.4)
    ax.set_yticks([0, .04, .08, .12])
    fs.clean(ax)
    ax.tick_params(axis="x", pad=4)
    return fs.save(fig, "practice", "ch09", "sample-efficiency")


if __name__ == "__main__":
    fig_flickr_bars()
    fig_wit_scaling()
    fig_sample_efficiency()
