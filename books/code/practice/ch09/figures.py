# -*- coding: utf-8 -*-
"""第 9 章插图：数据管线全流程、去重前后分布、recaption 流程与信息账、领域配比与温度采样。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs


# ---------------------------------------------------------------- 图 9-1
def fig_pipeline():
    fig = plt.figure(figsize=(7.2, 3.30))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    stages = [
        ("采集", ["URL 与元数据", "抓取时间"]),
        ("许可与安全", ["按来源分列", "先于混合"]),
        ("语言与长度", ["lid 打分", "切掉两端"]),
        ("质量与毒性", ["分类器打分", "阈值定尾部"]),
        ("模板与去重", ["去页眉页脚", "MinHash 分桶"]),
        ("配比与采样", ["域权重", "温度 T"]),
    ]
    n = len(stages)
    w = 0.128
    gap = (0.912 - n * w) / (n - 1)
    x0 = 0.040
    yb, hb = 0.800, 0.118

    for i, (name, notes) in enumerate(stages):
        x = x0 + i * (w + gap)
        fs.box(ax, x, yb, w, hb, name, fc="#FFFFFF", ec=fs.INK, fs=8.5, lw=1.1)
        fs.label(ax, x + w / 2, 0.762, notes[0], fs=7.0, color=fs.MUTED)
        fs.label(ax, x + w / 2, 0.722, notes[1], fs=7.0, color=fs.MUTED)
        if i:
            fs.arrow(ax, (x - gap + 0.004, yb + hb / 2), (x - 0.004, yb + hb / 2),
                     color=fs.FAINT, lw=0.95, ms=7)

    fs.label(ax, 0.040, 0.650, "同一条管线上的账面变化，图像侧取自 DINOv2 附录 A.3",
             fs=7.4, color=fs.MUTED, ha="left")
    ax.plot([0.040, 0.955], [0.618, 0.618], color=fs.GRID, lw=0.9, zorder=1)

    boxes = [("1.3B", "原始抓取", ""),
             ("1.1B", "自去重", "cos > 0.6，k = 64"),
             ("744M", "相对去重", "cos > 0.45"),
             ("142M", "检索后的训练集", "k = 4，近邻取样")]
    bw, y2, h2 = 0.172, 0.360, 0.140
    xs = [0.048, 0.286, 0.524, 0.762]
    for (num, note, sub), x in zip(boxes, xs):
        fs.box(ax, x, y2, bw, h2, "", fc="#F8FAFF", ec=fs.BLUE, lw=1.05)
        fs.label(ax, x + bw / 2, y2 + 0.090, num, fs=12.5, color=fs.INK, weight="bold")
        fs.label(ax, x + bw / 2, y2 + 0.038, note, fs=7.4, color=fs.MUTED)
        if sub:
            fs.label(ax, x + bw / 2, y2 - 0.030, sub, fs=6.9, color=fs.ACCENT)

    for i in range(3):
        xa = xs[i] + bw + 0.004
        xb = xs[i + 1] - 0.004
        fs.arrow(ax, (xa, y2 + 0.070), (xb, y2 + 0.070), color=fs.ACCENT, lw=1.0, ms=7)

    fs.label(ax, 0.048, 0.225,
             "两个去重步骤在删，检索一步在补，删掉的东西不会回到池子里。",
             fs=7.6, color=fs.INK, ha="left")
    fs.label(ax, 0.048, 0.150,
             "去重与防泄漏合起来拿走四成多，最后一步按目标分布选到 142M。",
             fs=7.6, color=fs.MUTED, ha="left")
    return fs.save(fig, "practice", "ch09", "pipeline")


# ---------------------------------------------------------------- 图 9-2
def fig_dedup():
    fig = plt.figure(figsize=(7.1, 3.30))
    ax = fig.add_axes([0.075, 0.175, 0.400, 0.720])
    ax2 = fig.add_axes([0.585, 0.175, 0.370, 0.720])

    # (a) 相似度分布，示意
    edges = np.linspace(0.0, 1.0, 41)
    cen = (edges[:-1] + edges[1:]) / 2
    before = 0.55 * np.exp(-cen / 0.18) + 1.35 * np.exp(-((cen - 0.93) / 0.055) ** 2)
    before += 0.30 * np.exp(-((cen - 0.62) / 0.14) ** 2)
    after = 0.52 * np.exp(-cen / 0.16)
    after += 0.02
    before = before / before.sum()
    after = after / after.sum()

    ax.bar(cen, before, width=0.021, color="#FBD5C0", edgecolor=fs.ACCENT,
           lw=0.55, label="去重前", zorder=3)
    ax.bar(cen, after, width=0.021, color="#C7D2FE", edgecolor=fs.BLUE,
           lw=0.55, label="去重后", zorder=4)

    for xv, lab in ((0.45, ".45"), (0.60, ".60")):
        ax.axvline(xv, color=fs.MUTED, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
        ax.text(xv + 0.012, 0.147, lab, fontsize=7.0, color=fs.MUTED, ha="left", va="top")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.150)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_yticks([])
    ax.set_xlabel("两两文档的 $J$", fontsize=8.4)
    ax.set_ylabel("相对频次", fontsize=8.4)
    fs.clean(ax, spines=("left", "bottom"))
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.015), ncol=2, fontsize=7.2,
                    handlelength=1.0, borderpad=0.25, columnspacing=1.2,
                    handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(fs.INK)
    ax.text(0.985, 0.870, "(a) 示意，非实测", transform=ax.transAxes, fontsize=7.2,
            color=fs.MUTED, ha="right", va="top")
    ax.annotate("镜像与转载", xy=(0.895, 0.090), xytext=(0.700, 0.104), fontsize=7.2,
                color=fs.ACCENT, ha="center",
                arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.7,
                                connectionstyle="arc3,rad=-0.20"))

    # (b) banding 的命中概率，精确计算
    J = np.linspace(0, 1, 400)
    specs = [(16, 8, fs.BLUE, "16 段 × 8 行（.71）"),
             (32, 4, fs.GREEN, "32 段 × 4 行（.42）"),
             (8, 16, fs.ACCENT, "8 段 × 16 行（.88）")]
    for b, r, c, lab in specs:
        P = 1 - (1 - J ** r) ** b
        thr = (1 / b) ** (1 / r)
        ax2.plot(J, P, color=c, lw=1.5, zorder=3, label=lab)
        ax2.plot([thr], [1 - (1 - thr ** r) ** b], marker="o", ms=4.2,
                 color=c, zorder=5, mec="white", mew=0.8)

    ax2.axhline(0.5, color=fs.GRID, lw=0.9, ls=(0, (2.2, 2.2)), zorder=1)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-0.02, 1.04)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels(["0", ".5", "1"], fontsize=8)
    ax2.set_xlabel("两段文本的 $J$", fontsize=8.4)
    ax2.set_ylabel("成为候选对的概率", fontsize=8.4)
    fs.clean(ax2, spines=("left", "bottom"))
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.930), fontsize=7.0,
               handlelength=1.3, borderpad=0.3, labelspacing=0.32)
    ax2.text(0.02, 0.975, "(b) 128 个哈希，按段分桶", transform=ax2.transAxes,
             fontsize=7.2, color=fs.MUTED, ha="left", va="top")
    return fs.save(fig, "practice", "ch09", "dedup")


# ---------------------------------------------------------------- 图 9-3
def fig_recaption():
    fig = plt.figure(figsize=(7.2, 3.45))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 左栏：流程
    fs.label(ax, 0.245, 0.960, "一条图片描述的去向", fs=8.4, color=fs.MUTED)
    fs.box(ax, 0.035, 0.775, 0.420, 0.125, "", fc="#FFF7ED", ec=fs.ACCENT, lw=1.05)
    fs.label(ax, 0.245, 0.865, "原描述（页面功能）", fs=7.4, color=fs.ACCENT)
    fs.label(ax, 0.245, 0.815, "Nike Air Max 90 黑白 正品 包邮", fs=8.0, color=fs.INK)

    fs.arrow(ax, (0.245, 0.770), (0.245, 0.665), color=fs.INK, lw=1.0)
    fs.label(ax, 0.262, 0.717, "看图重写", fs=7.2, color=fs.MUTED, ha="left")

    fs.box(ax, 0.035, 0.540, 0.420, 0.120, "", fc="#F8FAFF", ec=fs.BLUE, lw=1.05)
    fs.label(ax, 0.245, 0.622, "多模态描述模型", fs=8.4, color=fs.BLUE)
    fs.label(ax, 0.245, 0.575, "一次前向写一条，几十个词元", fs=7.2, color=fs.MUTED)

    fs.arrow(ax, (0.245, 0.535), (0.245, 0.470), color=fs.INK, lw=1.0)
    fs.box(ax, 0.035, 0.300, 0.420, 0.165, "", fc="#FFFFFF", ec=fs.INK, lw=1.05)
    fs.label(ax, 0.245, 0.435, "重写（视觉内容）", fs=7.4, color=fs.MUTED)
    fs.label(ax, 0.245, 0.390, "一双黑白配色的运动鞋放在灰色", fs=7.7, color=fs.INK)
    fs.label(ax, 0.245, 0.352, "地面上，侧面朝向镜头。", fs=7.7, color=fs.INK)

    fs.arrow(ax, (0.245, 0.295), (0.245, 0.225), color=fs.INK, lw=1.0)
    fs.box(ax, 0.035, 0.105, 0.420, 0.115, "", fc="#F3F4F6", ec=fs.FAINT, lw=1.0)
    fs.label(ax, 0.245, 0.185, "过滤：空话、幻觉、重复句式", fs=7.4, color=fs.INK)
    fs.label(ax, 0.245, 0.140, "留下的进入图文对", fs=7.4, color=fs.MUTED)

    ax.plot([0.500, 0.500], [0.075, 0.955], color=fs.GRID, lw=0.9, zorder=1)

    # 右栏：信息账
    fs.label(ax, 0.760, 0.960, "这笔账的四个方向", fs=8.4, color=fs.MUTED)
    rows = [("视觉内容的相关性", "↑", fs.BLUE, "描述真的在讲图里有什么"),
            ("写法的多样性", "↓", fs.ACCENT, "三条出自同一个模型，句式趋同"),
            ("专名、数字、地名", "↓", fs.ACCENT, "被抹成类别词，检索时对不上"),
            ("与另一个模型的耦合", "↑", fs.ACCENT, "对齐借了描述模型的视觉侧")]
    y = 0.845
    for name, sign, col, note in rows:
        fs.label(ax, 0.525, y, name, fs=8.0, color=fs.INK, ha="left")
        fs.label(ax, 0.955, y, sign, fs=11.0, color=col, ha="right", weight="bold")
        fs.label(ax, 0.525, y - 0.048, note, fs=7.0, color=fs.MUTED, ha="left")
        ax.plot([0.525, 0.955], [y - 0.086, y - 0.086], color=fs.GRID, lw=0.7, zorder=1)
        y -= 0.135

    fs.label(ax, 0.525, 0.240,
             "四个方向都是定性的，量级随模型与提示词变；",
             fs=7.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.525, 0.196,
             "只有第一条是想要的那一条。",
             fs=7.4, color=fs.INK, ha="left")
    fs.label(ax, 0.525, 0.135,
             "重写不是清洗，它换了一套干扰变量。",
             fs=7.6, color=fs.ACCENT, ha="left")
    return fs.save(fig, "practice", "ch09", "recaption")


# ---------------------------------------------------------------- 图 9-4
def fig_mixture():
    fig = plt.figure(figsize=(7.2, 3.35))
    ax = fig.add_axes([0.115, 0.190, 0.345, 0.690])
    ax2 = fig.add_axes([0.610, 0.190, 0.360, 0.690])

    p = np.array([0.90, 0.09, 0.01])
    names = ["通用图像", "文档截图", "专业细粒度"]
    cols = [fs.BLUE, fs.GREEN, fs.ACCENT]
    rows = [("自然占比", p), ("T = 0.5", None), ("T = 0.3", None)]
    for T, idx in ((0.5, 1), (0.3, 2)):
        q = p ** T
        rows[idx] = (rows[idx][0], q / q.sum())

    for i, (lab, q) in enumerate(rows[::-1]):
        left = 0.0
        for val, c in zip(q, cols):
            ax.barh(i, val, left=left, height=0.52, color=c, edgecolor="white",
                    lw=0.9, zorder=3)
            if val > 0.055:
                ax.text(left + val / 2, i, "%.0f%%" % (val * 100), ha="center", va="center",
                        fontsize=7.4, color="white", zorder=4)
            left += val
        ax.text(1.02, i, "%.0f / %.0f / %.0f" % (q[0] * 100, q[1] * 100, q[2] * 100),
                fontsize=6.9, color=fs.MUTED, va="center", ha="left")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel("采样时的占比", fontsize=8.4)
    fs.clean(ax, spines=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_color(fs.FAINT)
    hs = [Rectangle((0, 0), 1, 1, facecolor=c) for c in cols]
    leg = ax.legend(hs, names, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3,
                    fontsize=7.0, handlelength=0.9, columnspacing=1.1, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color(fs.INK)
    ax.text(0.0, -0.30, "(a) 三个域当作一个例子", transform=ax.transAxes,
            fontsize=7.2, color=fs.MUTED, ha="left")

    # (b) 提升倍数随 T 变化
    Ts = np.linspace(0.05, 1.0, 240)
    for val, c, lab in zip(p, cols, names):
        q = val ** Ts
        q = q / (p[0] ** Ts + p[1] ** Ts + p[2] ** Ts)
        ax2.plot(Ts, q / val, color=c, lw=1.5, zorder=3, label=lab)
    for val, c in zip(p, cols):
        q = val ** 0.5
        q = q / (p[0] ** 0.5 + p[1] ** 0.5 + p[2] ** 0.5)
        ax2.plot([0.5], [q / val], marker="o", ms=4.2, color=c, zorder=5,
                 mec="white", mew=0.8)
    ax2.axhline(1.0, color=fs.GRID, lw=1.0, zorder=1)
    ax2.text(0.065, 1.06, "不放大也不缩小", fontsize=7.0, color=fs.MUTED, ha="left", va="bottom")
    ax2.text(0.53, 9.2, "×7.4", fontsize=7.4, color=fs.ACCENT, ha="left", va="center")
    ax2.set_yscale("log")
    ax2.set_xlim(0.05, 1.0)
    ax2.set_ylim(0.3, 75)
    ax2.set_yticks([0.5, 1, 2, 5, 10, 20, 40])
    ax2.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "40"], fontsize=8)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax2.set_xlabel("温度 $T$", fontsize=8.4)
    ax2.set_ylabel("采样占比相对自然占比的倍数", fontsize=8.0)
    fs.clean(ax2, spines=("left", "bottom"))
    ax2.text(0.0, -0.30, "(b)", transform=ax2.transAxes, fontsize=8.4,
             color=fs.MUTED, ha="left", va="top")
    return fs.save(fig, "practice", "ch09", "mixture")


if __name__ == "__main__":
    fig_pipeline()
    fig_dedup()
    fig_recaption()
    fig_mixture()
