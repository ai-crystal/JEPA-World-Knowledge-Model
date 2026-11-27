# -*- coding: utf-8 -*-
"""第 10 章插图：三种并行的数据流、显存与通信分解、吞吐随批大小、扩展效率。

算例统一为 ViT-B/16：86M 参数、12 层、宽 768、12 头、patch 16，
224 分辨率下 197 个词元，每卡批大小按各图注明。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

# ---------------------------------------------------------------- 算例常数
N_PARAM = 86e6          # ViT-B/16 参数量
BYTES_BF16 = 2
# 参数侧显存（GB）
W_BF16 = N_PARAM * BYTES_BF16 / 1e9          # 0.172
W_FP32 = N_PARAM * 4 / 1e9                   # 0.344
ADAM = 2 * W_FP32                            # 0.688
GRAD = W_BF16                                # 0.172
OPT = W_FP32 + ADAM                          # 1.032
# 每张图激活（GB），c=16 份、L=12、T=197、d=768、bf16
ACT_PER_IMG = 16 * 12 * 197 * 768 * BYTES_BF16 / 1e9
# 检查点：只留 12 层输入（12 份 × 2d）+ 单块重算的临时峰值（约 4.8 MB）
ACT_CKPT = (12 * 197 * 768 * 2 * BYTES_BF16 / 1e9) + 4.85e-3


# ---------------------------------------------------------------- 图 10-1
def fig_parallel():
    fig = plt.figure(figsize=(7.1, 4.45))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    X0, X1 = 0.205, 0.975

    # ---- (a) 数据并行
    y = 0.688
    fs.label(ax, 0.012, 0.858, "(a) 数据并行", fs=9.6, weight="bold", ha="left")
    fs.label(ax, 0.012, 0.808, "分片对象：批", fs=8.0, color=fs.MUTED, ha="left")
    n = 8
    w = 0.083
    gap = (X1 - X0 - n * w) / (n - 1)
    for i in range(n):
        x = X0 + i * (w + gap)
        fs.box(ax, x, y, w, 0.085, "", fc="#EFF4FF", ec=fs.BLUE, lw=1.0)
        fs.label(ax, x + w / 2, y + 0.056, "%d" % i, fs=7.4, color=fs.MUTED)
        fs.label(ax, x + w / 2, y + 0.026, "整份模型", fs=7.4, color=fs.INK)
    fs.box(ax, X0, 0.540, X1 - X0, 0.072,
           "梯度全归约：172 MB 参数，每卡搬 301 MB",
           fc="#FFF7ED", ec=fs.ACCENT, fs=8.2, lw=1.0)
    fs.arrow(ax, (0.560, y - 0.006), (0.560, 0.618), color=fs.ACCENT, lw=1.0)
    fs.arrow(ax, (0.500, 0.618), (0.500, y - 0.006), color=fs.ACCENT, lw=1.0)
    fs.label(ax, X0, 0.822, "每卡一份完整副本，数据切八份", fs=7.4,
             color=fs.MUTED, ha="left")

    # ---- (b) 张量并行
    yb, bw, bh = 0.320, 0.200, 0.105
    fs.label(ax, 0.012, 0.462, "(b) 张量并行", fs=9.6, weight="bold", ha="left")
    fs.label(ax, 0.012, 0.412, "分片对象：层内矩阵", fs=8.0, color=fs.MUTED, ha="left")
    for k, bx in enumerate((0.205, 0.535)):
        fs.box(ax, bx, yb, bw, bh, "", fc="#FFFFFF", ec=fs.INK, lw=1.05)
        for j in range(8):
            sx = bx + 0.0115 + j * 0.0243
            ax.add_patch(plt.Rectangle((sx, yb + 0.026), 0.0185, bh - 0.052,
                                       facecolor="#DBEAFE" if j == 4 * k else "#F3F4F6",
                                       edgecolor=fs.FAINT, lw=0.5, zorder=4))
    fs.arrow(ax, (0.412, 0.392), (0.528, 0.392), color=fs.ACCENT, lw=1.0, ms=7)
    fs.arrow(ax, (0.528, 0.352), (0.412, 0.352), color=fs.ACCENT, lw=1.0, ms=7)
    fs.label(ax, 0.470, 0.424, "每块两次", fs=7.4, color=fs.ACCENT)
    fs.label(ax, 0.470, 0.326, "共 24 次", fs=7.4, color=fs.ACCENT)
    fs.label(ax, 0.470, 0.296, "每个矩阵切八份，各持一份", fs=7.6, color=fs.MUTED)
    fs.label(ax, 0.750, 0.386, "8 路时", fs=7.6, color=fs.MUTED, ha="left")
    fs.label(ax, 0.750, 0.354, "每卡搬 6.5 GB", fs=7.6, color=fs.ACCENT, ha="left")

    # ---- (c) 流水线并行
    yc = 0.105
    fs.label(ax, 0.012, 0.242, "(c) 流水线并行", fs=9.6, weight="bold", ha="left")
    fs.label(ax, 0.012, 0.192, "分片对象：层", fs=8.0, color=fs.MUTED, ha="left")
    stages = ["段 0\n层 0–2", "段 1\n层 3–5", "段 2\n层 6–8", "段 3\n层 9–11"]
    cw = 0.160
    xs = [X0 + i * (cw + 0.045) for i in range(4)]
    for x, s in zip(xs, stages):
        fs.box(ax, x, yc, cw, 0.080, s, fc="#EFF4FF", ec=fs.BLUE, fs=7.8, lw=1.0)
    for i in range(3):
        fs.arrow(ax, (xs[i] + cw + 0.005, yc + 0.058), (xs[i + 1] - 0.005, yc + 0.058),
                 color=fs.BLUE, lw=1.0, ms=6.5)
        fs.arrow(ax, (xs[i + 1] - 0.005, yc + 0.024), (xs[i] + cw + 0.005, yc + 0.024),
                 color=fs.ACCENT, lw=0.9, ms=6.5, ls=(0, (3.0, 1.6)))
    fs.label(ax, X0, 0.066, "每个微批 19 MB，实线前向、虚线反向", fs=7.4,
             color=fs.MUTED, ha="left")
    fs.label(ax, X1, 0.066, "气泡 27%（8 微批）", fs=7.4, color=fs.ACCENT, ha="right")
    return fs.save(fig, "practice", "ch10", "parallel")


# ---------------------------------------------------------------- 图 10-2
def fig_memory():
    fig = plt.figure(figsize=(7.2, 3.45))
    ax = fig.add_axes([0.080, 0.250, 0.430, 0.690])
    ax2 = fig.add_axes([0.655, 0.250, 0.310, 0.690])

    # ---- (a) 显存堆叠条
    bars = [
        ("批 128\n不重算", W_BF16, OPT, GRAD, 128 * ACT_PER_IMG),
        ("批 512\n不重算", W_BF16, OPT, GRAD, 512 * ACT_PER_IMG),
        ("批 512\n重算", W_BF16, OPT, GRAD, 512 * ACT_CKPT),
        ("批 2048\n不重算", W_BF16, OPT, GRAD, 2048 * ACT_PER_IMG),
        ("8 卡 ZeRO-3\n批 2048 重算", W_BF16 / 8, OPT / 8, GRAD / 8, 2048 * ACT_CKPT),
    ]
    cols = [fs.BLUE, fs.PURPLE, fs.GREEN, fs.ACCENT]
    names = ["bf16 权重", "fp32 主权重与 Adam", "bf16 梯度", "激活"]
    x = np.arange(len(bars))
    for i, (_, a, b, c, d) in enumerate(bars):
        bottom = 0.0
        for v, col in zip((a, b, c, d), cols):
            ax.bar(i, v, bottom=bottom, width=0.58, color=col, zorder=3,
                   edgecolor="white", linewidth=0.7)
            bottom += v
        ax.text(i, bottom + 2.5, "%.1f" % bottom, ha="center", va="bottom",
                fontsize=8.4, color=fs.INK, weight="bold")

    ax.axhline(80, color=fs.ACCENT, lw=1.0, ls=(0, (4, 2.4)), zorder=5)
    ax.text(4.42, 84, "A100 80 GB", fontsize=7.6, color=fs.ACCENT, ha="right")
    ax.annotate("", xy=(2.66, 80.8), xytext=(2.66, 117.5),
                arrowprops=dict(arrowstyle="<->", color=fs.ACCENT, lw=0.9,
                                shrinkA=1, shrinkB=1))
    ax.text(2.60, 99, "放不下", fontsize=7.6, color=fs.ACCENT, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.2)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_ylabel("每卡显存（GB）", fontsize=8.6)
    fs.clean(ax, spines=("left", "bottom"))
    ax.tick_params(axis="x", length=0)
    ax.set_title("（a）显存组成", fontsize=9.0, color=fs.MUTED, pad=24)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in cols]
    leg = ax.legend(handles, names, loc="upper left", fontsize=7.0, handlelength=1.0,
                    borderpad=0.2, labelspacing=0.35, ncol=2, columnspacing=0.8,
                    bbox_to_anchor=(0.0, 1.03))
    for t in leg.get_texts():
        t.set_color(fs.INK)

    # ---- (b) 每步通信量
    rows = [("数据并行\n梯度归约", 301, "步时的 3%"),
            ("流水线并行\n段间激活", 310, "步时的 11%")]
    mw = 155.0 * 2.0  # 每张量 155 MB，两次
    big = [("张量并行\n24 次归约", [3260, 3260], "步时的 30%")]
    yy = [0.70, 0.36, 0.0]
    for (lab, v, pct), y in zip(rows, yy[:2]):
        ax2.barh(y, v, height=0.30, color=fs.BLUE, zorder=3)
        ax2.text(v + 90, y, "%d MB" % v, va="center", ha="left", fontsize=7.8, color=fs.INK)
        ax2.text(v + 90, y - 0.115, pct, va="center", ha="left", fontsize=7.0, color=fs.MUTED)
    # 张量并行：两段堆叠（注意力块与 MLP 块）
    left = 0.0
    for v, col in zip(big[0][1], (fs.BLUE, fs.PURPLE)):
        ax2.barh(-0.34, v, left=left, height=0.30, color=col, zorder=3,
                 edgecolor="white", linewidth=0.7)
        left += v
    ax2.text(left - 110, -0.34, "6.5 GB，%s" % big[0][2], va="center", ha="right",
             fontsize=7.8, color="white", weight="bold")

    ax2.set_yticks([0.70, 0.36, -0.34])
    ax2.set_yticklabels([r[0] for r in rows] + [big[0][0]], fontsize=7.6)
    ax2.set_xlim(0, 7900)
    ax2.set_ylim(-0.66, 1.02)
    ax2.set_xticks([0, 2000, 4000, 6000])
    ax2.set_xticklabels(["0", "2", "4", "6 GB"], fontsize=7.6)
    ax2.set_xlabel("每卡每步搬运量（GB）", fontsize=8.4)
    fs.clean(ax2, spines=("bottom",))
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("（b）每卡每步通信量，全局批 4096", fontsize=9.0, color=fs.MUTED, pad=24)
    return fs.save(fig, "practice", "ch10", "memory")


# ---------------------------------------------------------------- 图 10-3
def fig_throughput():
    fig = plt.figure(figsize=(6.4, 3.30))
    ax = fig.add_axes([0.105, 0.175, 0.865, 0.760])

    B = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                  768, 1024, 1536, 2048], float)
    t0, c = 0.0125, 0.00078
    T = B / (t0 + c * B)
    decay = np.where(B > 512, 1 - 0.10 * np.log2(np.maximum(B, 512) / 512), 1.0)
    T = T * decay

    ax.plot(B, T, color=fs.BLUE, lw=1.7, zorder=4, marker="o", ms=3.2,
            markerfacecolor="white", markeredgewidth=1.0)

    # 三个区间
    ax.axvspan(8, 96, color=fs.LIGHT, zorder=0)
    ax.text(26, 1404, "延迟受限", fontsize=7.8, color=fs.MUTED, ha="center")
    ax.text(300, 1404, "算力受限", fontsize=7.8, color=fs.MUTED, ha="center")
    ax.text(1150, 1404, "带宽与显存受限", fontsize=7.6, color=fs.MUTED, ha="center")

    # 标注三个点
    for bb, txt, dx, dy in ((32, "853", 0, 62), (128, "1140", 10, 64),
                            (512, "1243", -34, 58), (2048, "1026", 0, -88)):
        v = float(B[B == bb][0])
        tv = float(T[B == bb][0])
        ax.plot([v], [tv], marker="o", ms=5.4, color=fs.ACCENT, zorder=6)
        ax.annotate(txt, xy=(v, tv), xytext=(v * (1 + dx / 100.0) if dx else v, tv + dy),
                    fontsize=7.8, color=fs.ACCENT, ha="center",
                    arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.7))

    # 渐进线
    ax.axhline(1282, color=fs.FAINT, lw=0.9, ls=(0, (1.8, 2.0)), zorder=1)
    ax.text(8.6, 1250, "虚线是每图 0.78 ms 的渐进值", fontsize=7.2, color=fs.MUTED, ha="left")

    ax.set_xscale("log")
    ax.set_xlim(7, 2600)
    ax.set_ylim(0, 1470)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512", "1024", "2048"],
                       fontsize=7.8)
    ax.set_yticks([0, 400, 800, 1200])
    ax.set_xlabel("每卡批大小（对数）", fontsize=8.6)
    ax.set_ylabel("每卡吞吐（张/秒）", fontsize=8.6)
    fs.clean(ax, spines=("left", "bottom"), grid=True)
    ax.grid(alpha=.55, lw=.6, zorder=0)
    ax.text(0.985, 0.035, "示意", transform=ax.transAxes, fontsize=8.2,
            color=fs.MUTED, ha="right")
    return fs.save(fig, "practice", "ch10", "throughput")


# ---------------------------------------------------------------- 图 10-4
def fig_scaling():
    fig = plt.figure(figsize=(6.2, 3.35))
    ax = fig.add_axes([0.115, 0.180, 0.855, 0.700])

    N = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048], float)
    dp = np.array([.95, .92, .88, .85, .82, .74, .62, .51, .40])
    tp = np.array([.86, .70, .55, .42, .33, np.nan, np.nan, np.nan, np.nan])

    ax.plot(N, np.ones_like(N), color=fs.FAINT, lw=1.0, ls=(0, (5, 3)), zorder=1)
    ax.text(1200, 1.015, "线性加速", fontsize=7.6, color=fs.MUTED, ha="right")

    ax.plot(N, dp, color=fs.BLUE, lw=1.8, marker="o", ms=4.2, zorder=4,
            markerfacecolor="white", markeredgewidth=1.1, label="纯数据并行")
    ax.plot(N[:5], tp[:5], color=fs.ACCENT, lw=1.8, marker="s", ms=4.0, zorder=4,
            markerfacecolor="white", markeredgewidth=1.1, label="8 路张量并行 + 数据并行")

    # 机内 / 跨机分界
    ax.axvline(8, color=fs.MUTED, lw=0.9, ls=(0, (2.4, 2.4)), zorder=2)
    ax.text(8.7, 1.075, "左边是一台八卡机内，往右就是跨机", fontsize=7.4, color=fs.MUTED,
            ha="left")

    for x, y in ((512, .62), (2048, .40)):
        ax.annotate("%.2f" % y, xy=(x, y), xytext=(x * 0.72, y + 0.075),
                    fontsize=7.8, color=fs.BLUE,
                    arrowprops=dict(arrowstyle="-", color=fs.BLUE, lw=0.7))
    ax.annotate("0.42", xy=(64, .42), xytext=(86, .53), fontsize=7.8, color=fs.ACCENT,
                ha="left", arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.7))

    ax.set_xscale("log", base=2)
    ax.set_xlim(6.4, 2800)
    ax.set_ylim(0, 1.16)
    ax.set_xticks([8, 32, 128, 512, 2048])
    ax.set_xticklabels(["8", "32", "128", "512", "2048"], fontsize=8)
    ax.set_yticks([0, .25, .50, .75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=8)
    ax.set_xlabel("加速卡数量", fontsize=8.6)
    ax.set_ylabel("扩展效率 $E(N)$", fontsize=8.6)
    fs.clean(ax, spines=("left", "bottom"), grid=True)
    ax.grid(alpha=.55, lw=.6, zorder=0)
    leg = ax.legend(loc="lower left", fontsize=7.6, handlelength=1.8, borderpad=0.25,
                    labelspacing=0.4)
    for t in leg.get_texts():
        t.set_color(fs.INK)
    ax.text(0.985, 0.035, "示意", transform=ax.transAxes, fontsize=8.2,
            color=fs.MUTED, ha="right")
    return fs.save(fig, "practice", "ch10", "scaling")


if __name__ == "__main__":
    print("每图激活 GB: %.3f, 检查点 %.3f" % (ACT_PER_IMG, ACT_CKPT))
    print("每卡批 128 显存: %.2f GB" % (W_BF16 + OPT + GRAD + 128 * ACT_PER_IMG))
    print("每卡批 512 显存: %.2f GB" % (W_BF16 + OPT + GRAD + 512 * ACT_PER_IMG))
    print("每卡批 2048 显存: %.2f GB" % (W_BF16 + OPT + GRAD + 2048 * ACT_PER_IMG))
    print("检查点 2048: %.2f GB" % (W_BF16 + OPT + GRAD + 2048 * ACT_CKPT))
    print("ZeRO-3 2048 检查点: %.2f GB" % ((W_BF16 + OPT + GRAD) / 8 + 2048 * ACT_CKPT))
    fig_parallel()
    fig_memory()
    fig_throughput()
    fig_scaling()
