# -*- coding: utf-8 -*-
"""第 1 章插图：下一词预测流程、文字作为世界的投影。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse
import figstyle as fs


def fig_next_token():
    fig = plt.figure(figsize=(7.0, 3.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 左侧：已有的词
    words = ["今天", "天气", "真"]
    ys = [0.78, 0.60, 0.42]
    for w, y in zip(words, ys):
        fs.box(ax, 0.03, y - 0.065, 0.105, 0.13, w, fc="#FFFFFF", ec=fs.BLUE,
               fs=9.5, lw=1.1)
        fs.arrow(ax, (0.137, y), (0.178, 0.60), color=fs.FAINT, lw=1.0, rad=0.04)
    fs.label(ax, 0.082, 0.93, "已读入的上下文", fs=8.6, color=fs.MUTED)

    # 中间：编码器
    fs.box(ax, 0.185, 0.34, 0.175, 0.36, "", fc="#F4F7FA", ec=fs.INK, lw=1.2)
    fs.label(ax, 0.2725, 0.585, "Transformer", fs=10, weight="bold")
    fs.label(ax, 0.2725, 0.505, "注意力 + 前馈", fs=8.2, color=fs.MUTED)
    fs.label(ax, 0.2725, 0.425, "共 N 层", fs=8.2, color=fs.MUTED)
    fs.label(ax, 0.2725, 0.265, "上下文向量", fs=8.4, color=fs.INK)

    fs.arrow(ax, (0.363, 0.52), (0.425, 0.52), color=fs.INK, lw=1.2)
    fs.box(ax, 0.428, 0.44, 0.10, 0.16, "输出层", fc="#FFFFFF", ec=fs.INK, fs=8.8, lw=1.1)
    fs.label(ax, 0.478, 0.392, "映射到整个词表", fs=7.6, color=fs.MUTED)
    fs.arrow(ax, (0.531, 0.52), (0.592, 0.52), color=fs.INK, lw=1.2)

    # 右侧：概率分布（示意，不标数值）
    axp = fig.add_axes([0.60, 0.30, 0.235, 0.50])
    cands = ["好", "不错", "热", "冷", "？"]
    vals = [0.62, 0.22, 0.09, 0.04, 0.03]
    cols = [fs.ACCENT] + [fs.FAINT] * 4
    axp.bar(range(5), vals, color=cols, width=0.62, zorder=3)
    axp.set_xticks(range(5)); axp.set_xticklabels(cands, fontsize=9.5)
    axp.set_yticks([])
    for s in ("top", "right", "left"):
        axp.spines[s].set_visible(False)
    axp.set_ylim(0, 0.78)
    axp.set_title("下一个词的概率", fontsize=8.6, color=fs.MUTED, pad=4)
    axp.annotate("", xy=(0, 0.65), xytext=(0, 0.78),
                 arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9))

    # 采样结果
    fs.box(ax, 0.865, 0.44, 0.115, 0.16, "好", fc="#FFF7F0", ec=fs.ACCENT, fs=11, lw=1.2)
    fs.arrow(ax, (0.838, 0.52), (0.863, 0.52), color=fs.ACCENT, lw=1.2, ms=10)

    # 回环
    ax.annotate("", xy=(0.082, 0.335), xytext=(0.082, 0.12),
                arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=1.0,
                                linestyle=(0, (4, 3))))
    ax.plot([0.082, 0.922], [0.12, 0.12], color=fs.FAINT, lw=1.0, ls=(0, (4, 3)))
    ax.plot([0.922, 0.922], [0.12, 0.40], color=fs.FAINT, lw=1.0, ls=(0, (4, 3)))
    fs.label(ax, 0.50, 0.055, "把新词接回输入，继续预测下一个", fs=8.4, color=fs.MUTED)
    return fs.save(fig, "basics", "ch01", "next-token")


def fig_shadow():
    fig = plt.figure(figsize=(6.6, 3.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    rows = [("物体类别", 1, 1), ("数量", 1, 1), ("粗略位置", 1, 1),
            ("精确几何形状", 1, 0), ("颜色与材质", 1, 0), ("遮挡关系", 1, 0),
            ("质量与受力", 0, 0), ("温度与气味", 0, 0)]
    x0, y0 = 0.06, 0.13
    rh = 0.085
    cw = [0.34, 0.16, 0.16]
    # 表头
    heads = ["场景中的信息", "照片", "文字描述"]
    xs = [x0, x0 + cw[0] + 0.01, x0 + cw[0] + cw[1] + 0.02]
    for h, x, w in zip(heads, xs, cw):
        fs.label(ax, x + w / 2, 0.925, h, fs=9.2, weight="bold", color=fs.INK)
    ax.plot([x0, x0 + sum(cw) + 0.02], [0.895, 0.895], color=fs.INK, lw=0.9)

    for i, (name, in_img, in_txt) in enumerate(rows):
        y = 0.86 - (i + 1) * rh
        if i % 2 == 0:
            ax.add_patch(Rectangle((x0 - 0.01, y + 0.004), sum(cw) + 0.03, rh - 0.008,
                                   facecolor="#F7F9FB", edgecolor="none", zorder=0))
        fs.label(ax, x0, y + rh / 2 - 0.004, name, fs=9, ha="left")
        for val, x, w in ((in_img, xs[1], cw[1]), (in_txt, xs[2], cw[2])):
            cx = x + w / 2
            if val:
                ax.scatter([cx], [y + rh / 2 - 0.004], s=42, color=fs.BLUE, zorder=3)
                ax.scatter([cx], [y + rh / 2 - 0.004], s=11, color="white", zorder=4)
            else:
                ax.scatter([cx], [y + rh / 2 - 0.004], s=42, facecolor="white",
                           edgecolor=fs.FAINT, linewidth=1.0, zorder=3)
                ax.plot([cx - 0.011, cx + 0.011], [y + rh / 2 - 0.016, y + rh / 2 + 0.008],
                        color=fs.FAINT, lw=1.0, zorder=4)

    ax.scatter([0.30], [0.045], s=42, color=fs.BLUE, zorder=3)
    ax.scatter([0.30], [0.045], s=11, color="white", zorder=4)
    fs.label(ax, 0.325, 0.045, "能被稳定记录", fs=8.4, color=fs.MUTED, ha="left")
    ax.scatter([0.58], [0.045], s=42, facecolor="white", edgecolor=fs.FAINT, linewidth=1.0, zorder=3)
    ax.plot([0.569, 0.591], [0.034, 0.056], color=fs.FAINT, lw=1.0, zorder=4)
    fs.label(ax, 0.605, 0.045, "记录时基本丢失", fs=8.4, color=fs.MUTED, ha="left")
    return fs.save(fig, "basics", "ch01", "shadow")


def fig_attn_cost():
    """注意力项与权重项随上下文长度的相对增长（按公式推算）。"""
    d = 4096                      # 隐藏维度，取一个常见的中等宽度
    n = np.logspace(np.log10(512), np.log10(262144), 400)
    n_cross = 4 * d               # 两项相当的位置
    lin = n / 512.0                       # 权重矩阵乘：与长度成正比
    quad = (n / n_cross) ** 2 * (n_cross / 512.0)   # 注意力：与长度平方成正比

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    fs.clean(ax, grid=False, spines=("left", "bottom"))
    ax.set_xscale("log"); ax.set_yscale("log")

    ax.plot(n, lin, color=fs.BLUE, lw=1.7, label="权重矩阵乘（与长度成正比）")
    ax.plot(n, quad, color=fs.ACCENT, lw=1.7, label="注意力打分与加权（与长度平方成正比）")
    ax.plot(n, lin + quad, color=fs.INK, lw=1.0, ls=(0, (5, 3)), label="两者之和")

    ax.plot([n_cross, n_cross], [0.5, 32], color=fs.FAINT, lw=0.9, ls=(0, (2, 3)))
    ax.scatter([n_cross], [32], s=26, color=fs.ACCENT, zorder=5,
               edgecolor="white", linewidth=0.8)
    ax.text(1000, 1100, "两项在隐藏维度的四倍附近相当", fontsize=8.2,
            color=fs.MUTED, ha="left", va="center")
    ax.annotate("d=4096 时约 1.6 万词元",
                xy=(n_cross, 32), xytext=(1000, 560),
                fontsize=8.2, color=fs.MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=fs.FAINT, lw=0.8,
                                connectionstyle="arc3,rad=-0.12"))

    ax.set_xlabel("上下文长度（词元）")
    ax.set_ylabel("处理一条序列的相对算力")
    ax.set_xticks([512, 2048, 8192, 32768, 131072])
    ax.set_xticklabels(["512", "2k", "8k", "32k", "128k"])
    ax.set_xlim(512, 262144)
    ax.set_ylim(0.5, 2.0e4)
    ax.legend(loc="upper left", fontsize=8.2, handlelength=1.8)
    ax.text(0.975, 0.30, "按公式推算，只画相对量级", transform=ax.transAxes,
            fontsize=8.2, color=fs.MUTED, ha="right")
    return fs.save(fig, "basics", "ch01", "attn-cost")


def fig_tokenization():
    """同一条中文句子在两种词表下的切分对比。"""
    large = ["我们", "把", "冰箱", "里的", "牛奶", "倒进", "杯子", "里"]
    small = list("我们把冰箱里的牛奶倒进杯子里")

    fig, ax = fs.canvas(6.9, 3.0)
    x0, span = 0.13, 0.80

    def row(tokens, ytop, h, fc, ec, fs_text):
        total = sum(len(t) for t in tokens)
        x = x0
        for i, t in enumerate(tokens):
            w = span * len(t) / total
            ax.add_patch(Rectangle((x, ytop - h), w, h, facecolor=fc, edgecolor=ec,
                                   lw=1.0, zorder=3))
            fs.label(ax, x + w / 2, ytop - h / 2, t, fs=fs_text,
                     color=fs.INK, z=4)
            x += w

    row(large, 0.86, 0.18, "#DBE7FF", fs.BLUE, 9.5)
    row(small, 0.52, 0.18, "#FDE6D8", fs.ACCENT, 8.8)

    fs.label(ax, x0, 0.925, "词表较大：8 个词元", fs=9, color=fs.BLUE, ha="left")
    fs.label(ax, x0, 0.585, "词表较小：14 个词元", fs=9, color=fs.ACCENT, ha="left")

    # 两行左右端对齐的提示
    for xv in (x0, x0 + span):
        ax.plot([xv, xv], [0.48, 0.90], color=fs.GRID, lw=0.9, ls=(0, (2, 3)), zorder=1)
    fs.label(ax, x0 + span / 2, 0.27, "同一句话，同样的十四个字",
             fs=8.6, color=fs.MUTED)
    fs.label(ax, x0 + span / 2, 0.16, "词元数差近一倍，后续每一步都要多算一遍",
             fs=8.6, color=fs.MUTED)
    return fs.save(fig, "basics", "ch01", "tokenization")


def report_numbers():
    """正文里出现的、需要由公式算出来的数字。"""
    d = 4096
    n_cross = 4 * d
    # 6ND 粗估：7B 参数、2 万亿词元
    N, D = 7e9, 2e12
    flops = 6 * N * D
    peak = 312e12            # A100 bf16 稠密算力峰值
    util = 0.40              # 按四成利用率计
    sec = flops / (peak * util)
    print(f"隐藏维度 {d} 时，两项相当的序列长度约 {n_cross} 词元")
    print(f"6ND = {flops:.2e} FLOPs；按 {peak/1e12:.0f} TFLOPS、利用率 {util:.0%}，"
          f"折合 {sec/86400:.0f} GPU 天（约 {sec/86400/365:.0f} GPU 年）")
    print(f"单卡若是 40 天，需要 {sec/86400/40:.0f} 张卡同时跑四十天")


if __name__ == "__main__":
    fig_next_token()
    fig_shadow()
    fig_attn_cost()
    fig_tokenization()
    report_numbers()
