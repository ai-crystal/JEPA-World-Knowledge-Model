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


if __name__ == "__main__":
    fig_next_token()
    fig_shadow()
