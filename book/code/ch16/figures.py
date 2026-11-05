# -*- coding: utf-8 -*-
"""第 16 章插图：Flickr 三组文本编码器、WIT 规模曲线、42 倍样本效率。

所有数值取自 Balestriero & LeCun, "Proving and Improving the Platonic
Representation Hypothesis with JEPAs"（arXiv:2610.39302）的 Table 1、
Table 2、Table 3 与 Figure 9 及其正文，不做任何插值或补点。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_flickr_bars():
    """图 16-1：Flickr30k 上图像 JEPA 对四种文本编码器的三条指标（Table 1）。"""
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
    bw = 0.155
    x0 = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for i, (name, vals, col) in enumerate(series):
        xs = x0 + (i - (n - 1) / 2) * (bw + 0.012)
        ax.bar(xs, vals, width=bw, color=col, label=name, zorder=3,
               edgecolor="white", linewidth=0.5)
        for x, v in zip(xs, vals):
            ax.text(x, max(v + 0.0045, 0.0075), f"{v:.3f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=7.4, color=fs.MUTED, zorder=4)

    ax.set_xticks(x0)
    ax.set_xticklabels([f"{m}\n随机基线 {c:.3f}".replace("0.", ".") for m, c in zip(metrics, chance)],
                       fontsize=9.2, color=fs.INK)
    ax.set_ylim(0, 0.163)
    ax.set_yticks([0, .04, .08, .12])
    ax.set_ylabel("指标数值（越高越好）", fontsize=9.4)
    fs.clean(ax, grid=True)
    ax.tick_params(axis="x", length=0, pad=6)

    ax.legend(loc="upper left", ncol=2, fontsize=8.6, handlelength=1.1,
              columnspacing=1.0, labelspacing=0.35, borderaxespad=0.4)
    ax.set_axisbelow(True)
    return fs.save(fig, "ch16", "flickr-bars")


def fig_wit_scaling():
    """图 16-2：WIT-1024 上的互 10-NN 随训练图片数上升（Figure 9 与正文）。"""
    m_good = [0.029, 2.16, 10, 50, 100]
    v_good = [.034, .108, .111, .125, .125]
    m_bad = [2.16, 50]
    v_bad = [.053, .063]
    m_ntp = [2.16, 50]
    v_ntp = [.071, .079]

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.set_xscale("log")
    ax.set_xlim(0.02, 700)
    ax.set_ylim(0.02, 0.142)

    # 语言模型参考线
    ax.axhline(.118, color=fs.GOLD, lw=1.0, ls=(0, (5, 3)), zorder=1,
               label="Qwen2.5-7B，.118")
    ax.axhline(.125, color=fs.PINK, lw=1.0, ls=(0, (5, 3)), zorder=1,
               label="Qwen2.5-3B，.125")

    ax.plot(m_bad, v_bad, color=fs.BLUE, lw=1.4, ls=(0, (4, 2)), marker="s",
            ms=4.5, zorder=4, markerfacecolor="white", markeredgewidth=1.1,
            label="坏视图 JEPA")
    ax.plot(m_ntp, v_ntp, color=fs.GREEN, lw=1.4, ls=(0, (2, 2)), marker="^",
            ms=5, zorder=4, markerfacecolor="white", markeredgewidth=1.1,
            label="下一词预测")
    ax.plot(m_good, v_good, color=fs.ACCENT, lw=1.9, marker="o", ms=6.2, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0, label="好视图 JEPA，33M")

    ax.set_xticks([0.03, 0.3, 2.16, 10, 50, 100])
    ax.set_xticklabels(["0.03\nFlickr 字幕", "0.3", "2.16", "10", "50", "100"])
    ax.set_xlabel("训练用到的图片数（百万，对数刻度）", fontsize=9.4, labelpad=7)
    ax.set_ylabel("与 DINOv2 ViT-B 的互 10-NN", fontsize=9.4)
    fs.clean(ax, grid=True)
    ax.tick_params(axis="x", pad=4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
              fontsize=8.5, handlelength=1.8, columnspacing=2.2,
              labelspacing=0.45, borderaxespad=0.0)
    return fs.save(fig, "ch16", "wit-scaling")


def fig_sample_efficiency():
    """图 16-3：同一个 109M 编码器达到同一对齐水平所需的样本数（正文 4.2 节）。"""
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.set_xscale("log")
    ax.set_xlim(8, 3000)
    ax.set_ylim(0, 0.148)

    pts = [(21.6, .109, fs.ACCENT, "好视图 JEPA"), (900, .111, fs.GREEN, "下一词预测")]

    for m, v, col, name in pts:
        ax.plot([m, m], [0, v], color=col, lw=0.9, ls=(0, (2, 3)), zorder=2)
        ax.plot([m], [v], marker="o", ms=10, color=col, zorder=5,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(m, v + .0075, f"{v:.3f}".replace("0.", "."), ha="center", va="bottom",
                fontsize=11, color=col, weight="bold")

    # 两个点之间直接连线，斜率就是 .002 的差别
    ax.plot([21.6, 900], [.109, .111], color=fs.FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.text(139, .119, "同一对齐水平", ha="center", va="bottom", fontsize=9,
            color=fs.MUTED)

    ax.annotate("", xy=(21.6, .040), xytext=(900, .040),
                arrowprops=dict(arrowstyle="<|-|>", color=fs.INK, lw=1.1,
                                mutation_scale=9))
    ax.text(139, .049, "样本少 42 倍", ha="center", va="bottom", fontsize=10.5,
            color=fs.INK)
    ax.text(139, .031, "词元少 14 倍，唯一图片少 435 倍", ha="center", va="top",
            fontsize=8.8, color=fs.MUTED)

    ax.set_xticks([10, 21.6, 100, 900, 3000])
    ax.set_xticklabels(["10", "21.6 百万\n好视图", "100", "900 百万\n下一词", "3000"])
    ax.set_xlabel("训练样本数（百万，对数刻度）", fontsize=9.4, labelpad=6)
    ax.set_ylabel("WIT-1024 上与 DINOv2 的互 10-NN", fontsize=9.4)
    ax.set_yticks([0, .04, .08, .12])
    fs.clean(ax)
    ax.tick_params(axis="x", pad=4)
    return fs.save(fig, "ch16", "sample-efficiency")


if __name__ == "__main__":
    fig_flickr_bars()
    fig_wit_scaling()
    fig_sample_efficiency()
