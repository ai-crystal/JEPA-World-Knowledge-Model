# -*- coding: utf-8 -*-
"""第 5 章插图：多块掩码、掩码消融的三条曲线、像素目标与表征目标的对比、
十二条复现清单、与 MAE 的算力对比。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs


# ---------------------------------------------------------------- 小工具
def grid_cells(ax, x0, y0, w, h, n, cell_fc, edge="white", lw=0.7, alpha=1.0, z=3):
    """在 (x0,y0,w,h) 里画 n x n 个方格，cell_fc 可以是单色或函数 f(i,j)->color。"""
    cw, ch = w / n, h / n
    for i in range(n):
        for j in range(n):
            fc = cell_fc(i, j) if callable(cell_fc) else cell_fc
            ax.add_patch(Rectangle((x0 + j * cw, y0 + (n - 1 - i) * ch), cw, ch,
                                   facecolor=fc, edgecolor=edge, lw=lw,
                                   alpha=alpha, zorder=z))


# ---------------------------------------------------------------- 图 5-1
BLOCKS = [  # (row_start, row_end_excl, col_start, col_end_excl, color)
    (0, 6, 0, 8, fs.ACCENT),
    (0, 7, 8, 15, fs.BLUE),
    (8, 15, 0, 6, fs.GREEN),
    (9, 15, 8, 15, fs.PURPLE),
]
N = 16


def _cell_index():
    """返回每个 patch 属于哪个目标块（-1 表示不属于任何块）。"""
    idx = -np.ones((N, N), int)
    for k, (r0, r1, c0, c1, _) in enumerate(BLOCKS):
        idx[r0:r1, c0:c1] = k
    return idx


def fig_mask_blocks():
    idx = _cell_index()
    ctx_r0, ctx_r1, ctx_c0, ctx_c1 = 0, 15, 0, 15          # 上下文大块，15x15
    in_ctx = np.zeros((N, N), bool)
    in_ctx[ctx_r0:ctx_r1, ctx_c0:ctx_c1] = True
    kept = in_ctx & (idx < 0)

    fig = plt.figure(figsize=(6.8, 2.95))
    axes = [fig.add_axes([0.030, 0.20, 0.255, 0.630]),
            fig.add_axes([0.372, 0.20, 0.255, 0.630]),
            fig.add_axes([0.714, 0.20, 0.255, 0.630])]
    for a in axes:
        a.set_xlim(0, N); a.set_ylim(0, N); a.axis("off")

    # (a) 原图：一个杯子的剪影
    a0 = axes[0]
    for i in range(N):
        for j in range(N):
            ci, cj = i + 0.5, j + 0.5
            d = np.hypot((ci - 8.0) / 4.6, (cj - 8.0) / 4.6)
            fc = "#C9B49C" if d < 1.0 else ("#E4D6C4" if d < 1.5 else "#F2EEE8")
            a0.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="white", lw=0.6))
    a0.set_xlim(0, N); a0.set_ylim(0, N)

    # (b) 目标块
    a1 = axes[1]
    for i in range(N):
        for j in range(N):
            k = idx[i, j]
            fc = "white" if k < 0 else BLOCKS[k][4]
            a1.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="white", lw=0.6,
                                   alpha=1.0 if k >= 0 else 0.85))
    for k, (r0, r1, c0, c1, col) in enumerate(BLOCKS):
        a1.add_patch(Rectangle((c0 + 0.04, N - r1 + 0.04), c1 - c0 - 0.08, r1 - r0 - 0.08,
                               facecolor="none", edgecolor=col, lw=1.3, zorder=5))
        ncell = (r1 - r0) * (c1 - c0)
        a1.text((c0 + c1) / 2, N - (r0 + r1) / 2, f"{ncell / N / N * 100:.0f}%",
                ha="center", va="center", fontsize=7.4, color=fs.INK,
                weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88))

    # (c) 上下文
    a2 = axes[2]
    for i in range(N):
        for j in range(N):
            if kept[i, j]:
                fc = "#BFDBFE"
            elif in_ctx[i, j]:
                fc = "#F7E2D0"
            else:
                fc = "#F4F4F5"
            a2.add_patch(Rectangle((j, N - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="#D4D4D8" if in_ctx[i, j] else "white",
                                   lw=0.6, zorder=3))
    a2.add_patch(Rectangle((ctx_c0 + 0.06, N - ctx_r1 + 0.06), ctx_c1 - ctx_c0 - 0.12,
                           ctx_r1 - ctx_r0 - 0.12, facecolor="none",
                           edgecolor=fs.INK, lw=1.1, ls=(0, (4, 3)), zorder=6))
    for k, (r0, r1, c0, c1, col) in enumerate(BLOCKS):
        a2.add_patch(Rectangle((c0 + 0.08, N - r1 + 0.08), c1 - c0 - 0.16, r1 - r0 - 0.16,
                               facecolor="none", edgecolor=col, lw=0.9,
                               ls=(0, (3, 2)), alpha=0.75, zorder=7))
    a2.text(2.6, 8.5, "可见", ha="center", va="center", fontsize=7.0, color="#1E3A8A",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75), zorder=8)
    a2.text(11.2, 13.2, "挖掉", ha="center", va="center", fontsize=7.0, color="#9A3412",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75), zorder=8)
    a2.text(N / 2, N + 0.6, "上下文大块 $15\\times15$", ha="center", va="bottom",
            fontsize=7.6, color=fs.INK)

    caps = ["（a）原图，切成 16 $\\times$ 16 个 patch",
            "（b）四个目标块，各占 16% 到 19%",
            f"（c）挖去目标区，留下 {kept.sum() / N / N * 100:.0f}% 可见"]
    for x, cap in zip([0.157, 0.500, 0.842], caps):
        fig.text(x, 0.085, cap, ha="center", va="center", fontsize=8.2, color=fs.INK)

    print("target ratios:", [f"{(r1-r0)*(c1-c0)/N/N*100:.1f}%" for r0, r1, c0, c1, _ in BLOCKS],
          "context kept:", f"{kept.sum() / N / N * 100:.1f}%")
    return fs.save(fig, "practice", "ch05", "mask-blocks")


# ---------------------------------------------------------------- 图 5-2
# 数据全部来自论文附录 Table 8 / 9 / 10：ViT-B/16，预训练 300 轮，
# 冻结主干后用 ImageNet-1K 的 1% 标签做线性探测，每次只改一个旋钮。
ABL_TARGET_SCALE = [("0.075–0.2", 19.2), ("0.10–0.2", 39.2), ("0.125–0.2", 42.4),
                    ("0.15–0.2", 54.2), ("0.20–0.25", 38.9), ("0.20–0.30", 33.6)]
ABL_CONTEXT_SCALE = [("0.40", 31.2), ("0.65", 47.1), ("0.75", 49.3), ("0.85", 54.2)]
ABL_TARGET_NUM = [("1", 9.0), ("2", 22.0), ("3", 48.5), ("4", 54.2)]


def _panel(ax, data, xlabel, note, mk_index, below=(), xtick_rot=0):
    ys = [v for _, v in data]
    xs = np.arange(len(data))
    ax.plot(xs, ys, color=fs.BLUE, lw=1.5, zorder=4, solid_capstyle="round")
    ax.scatter(xs, ys, s=26, color=fs.BLUE, zorder=5, linewidths=0)
    ax.scatter([xs[mk_index]], [ys[mk_index]], s=95, facecolor="white",
               edgecolor=fs.ACCENT, lw=1.5, zorder=6)
    ax.axhline(54.2, color=fs.FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for k, (x, y) in enumerate(zip(xs, ys)):
        if k in below:
            ax.text(x, y - 2.7, f"{y:.1f}", ha="center", va="top", fontsize=7.6,
                    color=fs.INK, zorder=7)
        else:
            ax.text(x, y + 2.6, f"{y:.1f}", ha="center", va="bottom", fontsize=7.6,
                    color=fs.INK, zorder=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([k for k, _ in data], fontsize=6.9,
                       rotation=xtick_rot, ha="right" if xtick_rot else "center")
    ax.set_xlim(-0.45, len(data) - 0.55)
    ax.set_ylim(0, 63)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel(xlabel, fontsize=8.2, labelpad=5)
    ax.tick_params(axis="y", labelsize=7.6, length=2.5, pad=2)
    ax.tick_params(axis="x", length=2.5, pad=2)
    fs.clean(ax)
    ax.text(0.02, 0.965, note, transform=ax.transAxes, fontsize=7.2,
            color=fs.MUTED, ha="left", va="top")


def fig_mask_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.05), sharey=True)

    _panel(axes[0], ABL_TARGET_SCALE, "目标块面积区间（占全图比例）",
           "上界固定 0.2", 3, below=(0,), xtick_rot=24)
    _panel(axes[1], ABL_CONTEXT_SCALE, "上下文面积下限",
           "目标块取 0.15–0.2", 3, below=(0,))
    _panel(axes[2], ABL_TARGET_NUM, "目标块个数",
           "上下文取 0.85–1.0", 3, below=(1,))

    axes[0].set_ylabel("1% 标签线性探测 top-1", fontsize=8.2, labelpad=3)
    for ax, tag in zip(axes, ["（a）目标块大小", "（b）上下文比例", "（c）目标块个数"]):
        ax.set_title(tag, fontsize=8.8, color=fs.INK, pad=6)

    fig.text(0.5, 0.018, "ViT-B/16，预训练 300 轮，冻结主干后只用 ImageNet-1K 的 1% 标签做线性探测",
             ha="center", va="bottom", fontsize=7.4, color=fs.MUTED)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.305, wspace=0.16)

    for name, data in [("target scale", ABL_TARGET_SCALE),
                       ("context scale", ABL_CONTEXT_SCALE),
                       ("target number", ABL_TARGET_NUM)]:
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in data))
    return fs.save(fig, "practice", "ch05", "mask-ablation")


# ---------------------------------------------------------------- 图 5-3
def _blob(n, noise=0.0, seed=0):
    """8x8 的示意图案，模拟一个物体的轮廓。"""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 0.90)
    for i in range(n):
        for j in range(n):
            d = ((i - 3.1) ** 2) / 3.4 + ((j - 3.6) ** 2) / 2.6
            if d < 1:
                img[i, j] = 0.42
            elif d < 1.7:
                img[i, j] = 0.72
    if noise:
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1)
    return img


def fig_pixel_vs_latent():
    fig = plt.figure(figsize=(7.0, 3.7))
    ax = fig.add_axes([0.0, 0.0, 0.695, 1.0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    n = 6
    gw = 0.155          # 网格在 x 方向上的宽度（y 方向高 0.21，按坐标轴长宽比换算）
    gh = 0.21

    # ---------- (a) 像素空间 ----------
    ax.text(0.020, 0.935, "（a）像素空间的目标", fontsize=8.8, color=fs.GOLD,
            weight="bold", ha="left", va="center")
    y_a = 0.64
    grid_cells(ax, 0.020, y_a, gw, gh, n, "#F0EBE3")
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]:
        cw = gw / n; ch = gh / n
        ax.add_patch(Rectangle((0.020 + j * cw, y_a + (n - 1 - i) * ch), cw, ch,
                               facecolor="#DFD2BF", edgecolor="white", lw=0.6, zorder=4))
    fs.label(ax, 0.020 + gw / 2, y_a - 0.035, "上下文块", fs=7.8, color=fs.MUTED)

    fs.box(ax, 0.215, y_a + 0.015, 0.100, gh - 0.03, "编码器", fc="white", ec=fs.INK,
           fs=8.2, lw=1.0)
    fs.box(ax, 0.345, y_a + 0.015, 0.100, gh - 0.03, "解码器", fc="white", ec=fs.GOLD,
           fs=8.2, lw=1.2)

    # 解码器输出（带噪声的像素）+ 原图像素
    out = _blob(n, noise=0.16, seed=3)
    tar = _blob(n, noise=0.0)
    cw = gw / n; ch = gh / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((0.485 + j * cw, y_a + (n - 1 - i) * ch), cw, ch,
                                   facecolor=str(out[i, j]), edgecolor="white", lw=0.6, zorder=4))
            ax.add_patch(Rectangle((0.705 + j * cw, y_a + (n - 1 - i) * ch), cw, ch,
                                   facecolor=str(tar[i, j]), edgecolor="white", lw=0.6, zorder=4))
    fs.label(ax, 0.485 + gw / 2, y_a - 0.035, "还原出的像素", fs=7.8, color=fs.GOLD)
    fs.label(ax, 0.705 + gw / 2, y_a - 0.035, "原图像素", fs=7.8, color=fs.MUTED)

    fs.arrow(ax, (0.180, y_a + gh / 2), (0.212, y_a + gh / 2), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.318, y_a + gh / 2), (0.342, y_a + gh / 2), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.448, y_a + gh / 2), (0.482, y_a + gh / 2), color=fs.MUTED, lw=1.0)
    ax.annotate("", xy=(0.700, y_a + gh / 2), xytext=(0.645, y_a + gh / 2),
                arrowprops=dict(arrowstyle="<->", color=fs.GOLD, lw=1.0))
    fs.label(ax, 0.672, y_a + gh + 0.045, "逐像素比", fs=7.4, color=fs.GOLD)

    # ---------- (b) 表征空间 ----------
    ax.text(0.020, 0.455, "（b）表征空间的目标", fontsize=8.8, color=fs.ACCENT,
            weight="bold", ha="left", va="center")
    y_b = 0.13
    grid_cells(ax, 0.020, y_b, gw, gh, n, "#F0EBE3")
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]:
        ax.add_patch(Rectangle((0.020 + j * (gw / n), y_b + (n - 1 - i) * (gh / n)),
                               gw / n, gh / n, facecolor="#DFD2BF", edgecolor="white",
                               lw=0.6, zorder=4))
    fs.label(ax, 0.020 + gw / 2, y_b - 0.035, "上下文块", fs=7.8, color=fs.MUTED)

    fs.box(ax, 0.215, y_b + 0.015, 0.100, gh - 0.03, "编码器", fc="white", ec=fs.INK,
           fs=8.2, lw=1.0)
    fs.box(ax, 0.345, y_b + 0.015, 0.100, gh - 0.03, "预测器", fc="white", ec=fs.ACCENT,
           fs=8.2, lw=1.2)
    fs.arrow(ax, (0.180, y_b + gh / 2), (0.212, y_b + gh / 2), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.318, y_b + gh / 2), (0.342, y_b + gh / 2), color=fs.MUTED, lw=1.0)
    fs.arrow(ax, (0.448, y_b + gh / 2), (0.482, y_b + gh / 2), color=fs.MUTED, lw=1.0)

    # 两条向量：预测 vs 目标
    rng = np.random.default_rng(7)
    va = np.clip(rng.normal(0.55, 0.18, 8), 0.1, 0.95)
    vb = np.clip(va + rng.normal(0, 0.09, 8), 0.1, 0.95)
    bw = 0.030
    for k in range(8):
        ax.add_patch(Rectangle((0.500, y_b + 0.004 + k * (gh - 0.008) / 8), bw,
                               (gh - 0.008) / 8, facecolor=str(va[k]),
                               edgecolor="white", lw=0.6, zorder=4))
        ax.add_patch(Rectangle((0.660, y_b + 0.004 + k * (gh - 0.008) / 8), bw,
                               (gh - 0.008) / 8, facecolor=str(vb[k]),
                               edgecolor="white", lw=0.6, zorder=4))
    fs.label(ax, 0.500 + bw / 2, y_b - 0.035, "预测 $\\hat{s}_y$", fs=7.8, color=fs.ACCENT)
    fs.label(ax, 0.660 + bw / 2, y_b - 0.035, "目标 $s_y$", fs=7.8, color=fs.MUTED)
    ax.annotate("", xy=(0.655, y_b + gh / 2), xytext=(0.535, y_b + gh / 2),
                arrowprops=dict(arrowstyle="<->", color=fs.ACCENT, lw=1.0))
    fs.label(ax, 0.595, y_b + gh + 0.045, "在向量上比", fs=7.4, color=fs.ACCENT)

    # ---------- (c) 柱状对比 ----------
    axb = fig.add_axes([0.760, 0.235, 0.225, 0.545])
    vals = [40.7, 66.9]
    axb.bar([0, 1], vals, color=[fs.FAINT, fs.ACCENT], width=0.55, zorder=3)
    for x, v in zip([0, 1], vals):
        axb.text(x, v + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=9.0,
                 color=fs.INK)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["像素目标", "表征目标"], fontsize=8.2)
    axb.set_ylim(0, 82)
    axb.set_yticks([0, 20, 40, 60, 80])
    axb.set_ylabel("1% 标签线性探测 top-1", fontsize=8.0, labelpad=3)
    axb.tick_params(labelsize=7.4)
    fs.clean(axb, spines=("left", "bottom"))
    axb.set_title("（c）两种目标下的探测精度", fontsize=8.2, color=fs.INK, pad=8)
    fig.text(0.872, 0.075, "ViT-L/16，同一套掩码", ha="center", va="center",
             fontsize=7.4, color=fs.MUTED)
    return fs.save(fig, "practice", "ch05", "pixel-vs-latent")


# ---------------------------------------------------------------- 图 5-4
# 十二条复现清单，按装配顺序分成四段。文字是本章的整理，不是论文原表的复述。
CHECKLIST = [
    ("① 数据", fs.BLUE, [
        (1, ["只用 ImageNet-1K", "训练集，不加颜色", "抖动与 mixup"]),
        (2, ["224 像素配 patch", "14，448 配 16，", "记下网格大小"]),
        (3, ["预训练不带 [cls]，", "评估靠 patch 的", "平均池化"]),
    ]),
    ("② 掩码", fs.ACCENT, [
        (4, ["目标 4 块，面积", "0.15–0.2，长宽比", "0.75–1.5"]),
        (5, ["上下文 0.85–1.0，", "长宽比 1，重叠", "部分整块挖掉"]),
        (6, ["同卡掩码长度取齐，", "采样器写进", "collator"]),
    ]),
    ("③ 结构", fs.PURPLE, [
        (7, ["预测器宽 384，深度", "按主干：B 6 层，", "L/H 12 层，G 16 层"]),
        (8, ["目标端复制初始化，", "动量 0.996 线性", "升到 1.0"]),
        (9, ["目标侧 detach，", "掩码打在目标端", "输出上"]),
    ]),
    ("④ 优化与评估", fs.GOLD, [
        (10, ["AdamW，batch 2048，", "1e-4 升到 1e-3，", "预热 15 轮，", "余弦退到 1e-6"]),
        (11, ["权重衰减 0.04", "线性升到 0.4"]),
        (12, ["评估用目标编码器，", "报数带轮数、", "分辨率与协议"]),
    ]),
]


def fig_checklist():
    fig, ax = fs.canvas(7.2, 2.92)
    n = len(CHECKLIST)
    gap, x0 = 0.026, 0.008
    w = (1 - 2 * x0 - gap * (n - 1)) / n
    y0, h = 0.02, 0.95
    for i, (title, color, items) in enumerate(CHECKLIST):
        x = x0 + i * (w + gap)
        fs.box(ax, x, y0, w, h, "", fc="#FFFFFF", ec=color, lw=1.1, rounded=0.012)
        ax.add_patch(Rectangle((x + 0.005, y0 + h - 0.125), w - 0.010, 0.105,
                               facecolor=color, alpha=0.11, lw=0, zorder=4))
        fs.label(ax, x + w / 2, y0 + h - 0.072, title, fs=9.0, color=color, weight="bold")
        yy = y0 + h - 0.155
        for num, lines in items:
            fs.label(ax, x + 0.013, yy, str(num), fs=7.6, color=color, weight="bold",
                     ha="left", va="top")
            fs.label(ax, x + 0.048, yy, "\n".join(lines), fs=6.8, color=fs.INK,
                     ha="left", va="top")
            yy -= 0.0605 * len(lines) + 0.042
        if i:
            fs.arrow(ax, (x - gap + 0.002, y0 + h - 0.072), (x - 0.003, y0 + h - 0.072),
                     color=fs.FAINT, lw=1.0, ms=7)
    return fs.save(fig, "practice", "ch05", "checklist")


# ---------------------------------------------------------------- 图 5-5
# (a)(b) 取自论文第 7 节的 7% 与 5 倍；(c) 的 1200 卡时取自摘要，
# 另两根柱按 2.5 倍与 10 倍下界换算，是下限而非实测。
def fig_compute_compare():
    fig = plt.figure(figsize=(7.2, 2.75))

    # (a) 单步耗时
    ax1 = fig.add_axes([0.072, 0.20, 0.175, 0.56])
    vals = [1.00, 1.07]
    ax1.bar([0, 1], vals, color=[fs.FAINT, fs.ACCENT], width=0.52, zorder=3)
    for x, v in zip([0, 1], vals):
        ax1.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8.2,
                 color=fs.INK)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["MAE", "I-JEPA"], fontsize=8.0)
    ax1.set_ylim(0, 1.28); ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_ylabel("单步耗时（相对值）", fontsize=8.0, labelpad=3)
    ax1.tick_params(labelsize=7.4, length=2.5, pad=2)
    fs.clean(ax1)
    ax1.text(0.5, 1.20, "+7%", ha="center", va="center", fontsize=8.0, color=fs.ACCENT,
             weight="bold")
    ax1.set_title("（a）一次迭代", fontsize=8.8, color=fs.INK, pad=7)

    # (b) 收敛所需轮数
    ax2 = fig.add_axes([0.352, 0.20, 0.175, 0.56])
    ep = [1600, 300]
    ax2.bar([0, 1], ep, color=[fs.FAINT, fs.ACCENT], width=0.52, zorder=3)
    for x, v in zip([0, 1], ep):
        ax2.text(x, v + 40, f"{v}", ha="center", va="bottom", fontsize=8.2, color=fs.INK)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["MAE", "I-JEPA"], fontsize=8.0)
    ax2.set_ylim(0, 1980); ax2.set_yticks([0, 500, 1000, 1500])
    ax2.set_ylabel("所需轮数（epoch）", fontsize=8.0, labelpad=3)
    ax2.tick_params(labelsize=7.4, length=2.5, pad=2)
    fs.clean(ax2)
    ax2.text(0.5, 1830, "5.3 倍", ha="center", va="center", fontsize=8.0, color=fs.ACCENT,
             weight="bold")
    ax2.set_title("（b）迭代次数", fontsize=8.8, color=fs.INK, pad=7)

    # (c) 总卡时，对数横轴
    ax3 = fig.add_axes([0.660, 0.235, 0.325, 0.525])
    rows = [("MAE  ViT-H/14", 12000, fs.FAINT, "≥12 000"),
            ("iBOT  ViT-S/16", 3000, fs.MUTED, "≥3 000"),
            ("I-JEPA  ViT-H/14", 1200, fs.ACCENT, "≤1 200")]
    xlo = 600.0
    for k, (name, v, col, tag) in enumerate(rows):
        ax3.barh(k, v - xlo, left=xlo, height=0.46, color=col, zorder=3)
        ax3.text(v * 1.10, k, tag, ha="left", va="center", fontsize=7.8, color=fs.INK)
    ax3.axvline(1200, color=fs.ACCENT, lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax3.set_yticks(range(len(rows)))
    ax3.set_yticklabels([r[0] for r in rows], fontsize=7.8)
    ax3.set_xscale("log")
    ax3.set_xlim(xlo, 42000)
    ax3.set_xticks([1000, 3000, 10000, 30000])
    ax3.set_xticklabels(["1 千", "3 千", "1 万", "3 万"])
    ax3.set_xlabel("预训练总卡时（A100）", fontsize=8.0, labelpad=4)
    ax3.tick_params(labelsize=7.4, length=2.5, pad=2, which="major")
    ax3.minorticks_off()
    fs.clean(ax3)
    ax3.invert_yaxis()
    ax3.set_title("（c）预训练总成本", fontsize=8.8, color=fs.INK, pad=7)

    fig.text(0.660, 0.045, "MAE、iBOT 两柱为下界换算值，非实测",
             ha="left", va="bottom", fontsize=7.0, color=fs.MUTED)
    return fs.save(fig, "practice", "ch05", "compute-compare")


def report():
    """正文引用的派生量：都由本脚本算出，不靠外部来源。"""
    n224_14 = (224 // 14) ** 2
    n448_16 = (448 // 16) ** 2
    print(f"  224/patch14: {224 // 14}×{224 // 14} = {n224_14} patch")
    print(f"  448/patch16: {448 // 16}×{448 // 16} = {n448_16} patch")
    print(f"  0.15 面积折合: {0.15 * n224_14:.0f} 与 {0.15 * n448_16:.0f} patch")
    print(f"  patch 数倍数: {n448_16 / n224_14:.2f}")
    print(f"  batch 2048 / 16 卡: {2048 // 16}")
    print(f"  1600/600 = {1600 / 600:.1f}, 1600/300 = {1600 / 300:.1f}")
    for N, p in [(224, 14), (384, 16), (448, 16)]:
        g = N // p
        r = g / 16.0
        print(f"  {N}/patch{p}: 网格 {g}×{g} = {g * g} patch, "
              f"保 patch 数 s'={0.15 / r ** 2:.3f}–{0.2 / r ** 2:.3f}, "
              f"保像素面积 s'={0.15 / (N / 224) ** 2:.3f}–{0.2 / (N / 224) ** 2:.3f}")
    print(f"  注意力成本比 (28/16)^4 = {(28 / 16) ** 4:.2f}")
    print(f"  16 卡 × 72 小时 = {16 * 72} 卡时")


if __name__ == "__main__":
    report()
    fig_mask_blocks()
    fig_mask_ablation()
    fig_pixel_vs_latent()
    fig_checklist()
    fig_compute_compare()
