# -*- coding: utf-8 -*-
"""第 7 章插图：世界模型闭环、温度与迁移分数、Dreamer 的隐空间想象。

数字来自 papers/1803.10122（Table 1、Table 2）与
papers/2301.04104（RSSM 与想象训练的超参）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_loop():
    """图 7-1 世界模型闭环。"""
    fig = plt.figure(figsize=(6.9, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0.085, 0.995); ax.axis("off")

    # 第一行：环境 → 编码器 → 状态 → 动力学
    y1, h = 0.66, 0.15
    fs.box(ax, 0.045, y1, 0.150, h, "环境\n真实世界", fc="#FBF7F0", ec=fs.INK, fs=9.2)
    fs.box(ax, 0.245, y1, 0.125, h, "编码器\n$E$", fc="#FFFFFF", ec=fs.BLUE, fs=9.2, lw=1.1)
    fs.box(ax, 0.415, y1, 0.125, h, "状态\n$s_t$", fc="#F2F6FF", ec=fs.BLUE, fs=9.4, lw=1.2)
    fs.box(ax, 0.585, y1, 0.165, h, "动力学\n$f_\\theta(s_t,a_t)$", fc="#FFFFFF",
           ec=fs.ACCENT, fs=9.2, lw=1.2)

    fs.arrow(ax, (0.197, 0.735), (0.242, 0.735), color=fs.INK, lw=1.1)
    fs.arrow(ax, (0.372, 0.735), (0.412, 0.735), color=fs.INK, lw=1.1)
    fs.arrow(ax, (0.542, 0.735), (0.582, 0.735), color=fs.INK, lw=1.1)
    fs.label(ax, 0.219, 0.618, "观测 $x_t$", fs=8.4, color=fs.MUTED)
    fs.label(ax, 0.392, 0.618, "压成向量", fs=8.4, color=fs.MUTED)
    fs.label(ax, 0.562, 0.618, "预测下一状态", fs=8.4, color=fs.MUTED)

    # 预测结果回灌：在内部连续推进
    fs.arrow(ax, (0.735, 0.815), (0.492, 0.818), color=fs.ACCENT, lw=1.1,
             ls=(0, (4, 3)), rad=0.42, ms=9, z=6)
    fs.label(ax, 0.605, 0.955, "把预测值喂回，在内部连续推进", fs=8.2, color=fs.ACCENT)

    # 第二行：策略 → 动作
    y2 = 0.38
    fs.box(ax, 0.415, y2, 0.125, h, "策略\n$\\pi$", fc="#FFFFFF", ec=fs.GREEN, fs=9.4, lw=1.1)
    fs.box(ax, 0.585, y2, 0.165, h, "动作\n$a_t$", fc="#F0FBF6", ec=fs.GREEN, fs=9.4, lw=1.2)
    fs.arrow(ax, (0.4775, 0.658), (0.4775, 0.533), color=fs.INK, lw=1.1)
    fs.arrow(ax, (0.542, 0.455), (0.582, 0.455), color=fs.INK, lw=1.1)
    fs.label(ax, 0.4875, 0.572, "读状态", fs=8.4, color=fs.MUTED, ha="left")
    fs.arrow(ax, (0.6675, 0.533), (0.6675, 0.658), color=fs.ACCENT, lw=1.0,
             ls=(0, (3, 3)), ms=8)
    fs.label(ax, 0.684, 0.596, "作为条件输入", fs=8.2, color=fs.ACCENT, ha="left")

    # 动作执行后回到环境
    ax.plot([0.6675, 0.6675], [0.378, 0.215], color=fs.FAINT, lw=1.0, zorder=2)
    ax.plot([0.6675, 0.120], [0.215, 0.215], color=fs.FAINT, lw=1.0, zorder=2)
    fs.arrow(ax, (0.120, 0.215), (0.120, 0.658), color=fs.MUTED, lw=1.1, ms=10)
    fs.label(ax, 0.395, 0.145, "执行动作，环境返回下一帧观测 $x_{t+1}$", fs=8.4, color=fs.MUTED)
    return fs.save(fig, "basics", "ch07", "world-model-loop")


def fig_temperature():
    """图 7-2 温度与分数：梦里的分数和搬回真实场景的分数。"""
    tau = np.array([0.10, 0.50, 1.00, 1.15, 1.30])
    virtual = np.array([2086, 2060, 1145, 918, 732])
    vsd = np.array([140, 277, 690, 546, 269])
    actual = np.array([193, 196, 868, 1092, 753])
    asd = np.array([58, 50, 511, 556, 139])
    x = np.arange(len(tau))

    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.155, top=0.885)
    fs.clean(ax, spines=("left", "bottom"))
    ax.set_ylim(0, 2450)
    ax.set_xlim(-0.55, 4.65)

    # 参考线：解出标准、随机策略
    ax.axhline(750, color=fs.FAINT, lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.axhline(210, color=fs.FAINT, lw=0.9, ls=(0, (1, 3)), zorder=1)
    ax.text(-0.50, 790, "解出标准 750", fontsize=8.2, color=fs.MUTED,
            ha="left", va="bottom")
    ax.text(4.60, 178, "随机策略 210", fontsize=8.2, color=fs.MUTED,
            ha="right", va="top")

    ax.errorbar(x - 0.055, virtual, yerr=vsd, color=fs.BLUE, lw=1.6, marker="o",
                ms=5.5, capsize=3.2, elinewidth=0.9, label="在虚拟环境里测", zorder=4)
    ax.errorbar(x + 0.055, actual, yerr=asd, color=fs.ACCENT, lw=1.6, marker="s",
                ms=5.0, capsize=3.2, elinewidth=0.9, label="搬回真实环境测", zorder=4)

    # 关键读数
    kw = dict(fontsize=8.4, va="center")
    ax.text(x[0] - 0.14, 2120, "2086", color=fs.BLUE, ha="right", **kw)
    ax.text(x[0] + 0.14, 128, "193", color=fs.ACCENT, ha="left", **kw)
    ax.text(x[3] - 0.13, 858, "918", color=fs.BLUE, ha="right", **kw)
    ax.text(x[3] + 0.14, 1195, "1092", color=fs.ACCENT, ha="left", **kw)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in tau], fontsize=9.2)
    ax.set_xlabel("采样温度 $\\tau$", fontsize=9.4, color=fs.INK, labelpad=5)
    ax.set_ylabel("存活步数（100 局平均）", fontsize=9.4, color=fs.INK, labelpad=6)
    ax.tick_params(axis="y", labelsize=9)

    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, fontsize=8.6,
              handlelength=2.0, columnspacing=1.6)
    return fs.save(fig, "basics", "ch07", "temperature-transfer")


def fig_imagine():
    """图 7-3 真实轨迹与想象轨迹。"""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.35))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.855, wspace=0.16)

    for ax in axes:
        ax.set_xlim(0.2, 8.6)
        ax.set_ylim(0.4, 9.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("潜空间的第一维（示意）", fontsize=8.4, color=fs.MUTED)
        ax.set_ylabel("第二维", fontsize=8.4, color=fs.MUTED)
        fs.clean(ax, spines=("left", "bottom"))

    # (a) 真实轨迹：编码器对回放片段逐帧编码
    a = axes[0]
    real = np.array([[0.9, 2.0], [2.1, 2.7], [3.5, 3.2], [4.9, 4.1], [6.0, 5.0]])
    a.plot(real[:, 0], real[:, 1], color=fs.BLUE, lw=1.8, zorder=3, solid_capstyle="round")
    for i in range(len(real) - 1):
        p0, p1 = real[i], real[i + 1]
        d = p1 - p0
        fs.arrow(a, tuple(p0 + d * 0.62), tuple(p0 + d * 0.80), color=fs.BLUE, lw=1.2, ms=9)
    a.scatter(real[:, 0], real[:, 1], s=46, color=fs.BLUE, zorder=4,
              edgecolor="white", linewidth=1.2)
    for i, (xx, yy) in enumerate(real):
        a.text(xx, yy - 0.62, f"$s_{i + 1}$", fontsize=8.8, color=fs.BLUE,
               ha="center", va="top", zorder=5)
    a.set_title("（a）回放缓冲里的真实片段", fontsize=9.4, color=fs.INK, pad=8)
    a.text(0.32, 8.35, "观测逐帧过编码器", fontsize=8.4, color=fs.MUTED, ha="left")

    # (b) 想象轨迹：从同一状态出发，策略采样动作，动力学推进
    b = axes[1]
    start = np.array([1.6, 4.4])
    paths = [
        (fs.ACCENT, [[2.7, 5.2], [3.9, 5.7], [5.1, 6.3], [6.3, 6.7], [7.4, 7.4]]),
        (fs.PURPLE, [[2.8, 4.0], [4.0, 3.4], [5.2, 3.0], [6.4, 2.5], [7.5, 2.4]]),
        (fs.GREEN,  [[2.6, 4.5], [3.8, 4.6], [5.0, 4.1], [6.2, 4.7], [7.3, 4.9]]),
    ]
    for col, pts in paths:
        p = np.vstack([start, np.array(pts)])
        b.plot(p[:, 0], p[:, 1], color=col, lw=1.4, ls=(0, (4, 2.5)), zorder=3)
        b.scatter(p[1:, 0], p[1:, 1], s=20, facecolor="white", edgecolor=col,
                  linewidth=1.1, zorder=4)
        b.scatter([p[-1, 0]], [p[-1, 1]], s=32, color=col, zorder=4,
                  edgecolor="white", linewidth=1.1)
    b.scatter([start[0]], [start[1]], s=64, color=fs.INK, zorder=5,
              edgecolor="white", linewidth=1.3)
    b.text(start[0] - 0.14, start[1] + 0.55, "$s_t$", fontsize=9.4, color=fs.INK, ha="left")
    b.set_title("（b）从同一状态分叉的想象", fontsize=9.4, color=fs.INK, pad=8)
    b.text(0.32, 8.35, "同一状态、不同动作序列", fontsize=8.4, color=fs.MUTED, ha="left")
    b.annotate("", xy=(7.5, 1.78), xytext=(2.6, 1.78),
               arrowprops=dict(arrowstyle="<->", color=fs.FAINT, lw=0.9))
    b.text(5.05, 1.05, "在动力学里推进 $T=16$ 步", fontsize=8.4, color=fs.MUTED, ha="center", va="center")
    return fs.save(fig, "basics", "ch07", "dreamer-imagination")


def fig_three_world_models():
    """图 7-4 三类世界模型的输入、状态、输出。

    World Models 的 32 维与 64×64 见 papers/1803.10122；
    Genie 的 11B 参数与 3 万小时视频见 papers/2402.15391；
    RSSM 的 h_t 与 z_t 见 papers/2301.04104 式 (1)。
    """
    fig = plt.figure(figsize=(6.9, 4.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    cols = [(0.185, 0.200), (0.425, 0.195), (0.655, 0.215)]
    for (x, w), hd in zip(cols, ["输入", "状态", "输出"]):
        fs.label(ax, x + w / 2, 0.930, hd, fs=9.6, color=fs.MUTED, weight="bold")
    # 列头下面的短横线
    for x, w in cols:
        ax.plot([x + 0.02, x + w - 0.02], [0.900, 0.900], color=fs.GRID, lw=1.0, zorder=1)

    h = 0.200
    rows = [
        (0.655, "像素生成式", "World Models\nGenie", "#FDF6EC", fs.GOLD),
        (0.395, "潜空间预测式", "JEPA\nV-JEPA 2", "#F2F6FF", fs.BLUE),
        (0.135, "混合式", "Dreamer\nRSSM", "#F7F3FD", fs.PURPLE),
    ]
    texts = [
        ["观测帧 $x_t$\n加动作 $a_t$", "像素本身\n或视觉 token", "下一帧画面 $x_{t+1}$\n能播给人看"],
        ["观测帧 $x_t$\n加动作 $a_t$", "特征向量 $s_t$\n编码器输出", "预测向量 $\\hat{s}_{t+1}$\n不解码回画面"],
        ["观测帧 $x_t$\n加动作 $a_t$", "$h_t$ 与 $z_t$\n确定一半＋随机一半", "推进 $T$ 步的潜状态\n训练时才解码"],
    ]
    for (y, name, note, fc, ec), tx in zip(rows, texts):
        ax.text(0.018, y + h / 2 + 0.026, name, fontsize=9.4, color=ec,
                weight="bold", ha="left", va="center", zorder=5)
        ax.text(0.018, y + h / 2 - 0.038, note, fontsize=8.0, color=fs.MUTED,
                ha="left", va="center", linespacing=1.45, zorder=5)
        for (x, w), t in zip(cols, tx):
            fs.box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=8.6, lw=1.1)
        yc = y + h / 2
        fs.arrow(ax, (cols[0][0] + cols[0][1] + 0.008, yc), (cols[1][0] - 0.008, yc),
                 color=fs.MUTED, lw=1.0, ms=8)
        fs.arrow(ax, (cols[1][0] + cols[1][1] + 0.008, yc), (cols[2][0] - 0.008, yc),
                 color=fs.MUTED, lw=1.0, ms=8)
    return fs.save(fig, "basics", "ch07", "three-world-models")


def fig_rollout_error():
    """图 7-5 误差随时域增长（示意，无实测数值）。

    只画出方向与三条曲线的相对高低，纵轴不标刻度；
    16 步取自 papers/2301.04104 的想象长度 T = 16。
    """
    t = np.arange(0.0, 41.0, 0.25)
    pix = 0.06 + 0.92 * (1.0 - np.exp(-t / 5.5))
    lat = 0.05 + 0.62 * (1.0 - np.exp(-t / 11.0))
    spread = 0.055 * (1.0 - np.exp(-t / 9.0))
    real = np.full_like(t, 0.035)

    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.150, top=0.845)
    fs.clean(ax, spines=("left", "bottom"))
    ax.set_xlim(-1.5, 57.0)
    ax.set_ylim(0.0, 1.16)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("从当前状态往前推的步数", fontsize=9.4, color=fs.INK, labelpad=6)
    ax.set_ylabel("预测误差（示意，不标刻度）", fontsize=9.4, color=fs.INK, labelpad=8)

    ax.fill_between(t, lat - spread, lat + spread, color=fs.BLUE, alpha=0.12,
                    linewidth=0, zorder=1)
    ax.plot(t, pix, color=fs.GOLD, lw=1.9, zorder=4)
    ax.plot(t, lat, color=fs.BLUE, lw=1.9, zorder=4)
    ax.plot(t, real, color=fs.GREEN, lw=1.6, ls=(0, (4, 2.5)), zorder=4)

    # 线端文字，放在坐标区右侧的留白里
    ax.text(42.0, pix[-1] + 0.005, "像素重建", fontsize=9.0, color=fs.GOLD,
            ha="left", va="center")
    ax.text(42.0, lat[-1] + 0.005, "潜空间预测", fontsize=9.0, color=fs.BLUE,
            ha="left", va="center")
    ax.text(42.0, real[-1] + 0.045, "真实执行", fontsize=9.0, color=fs.GREEN,
            ha="left", va="center")

    # 16 步：Dreamer 的想象长度
    ax.axvline(16, color=fs.FAINT, lw=0.9, ls=(0, (5, 3)), zorder=2)
    ax.text(16.9, 1.085, "Dreamer 的想象长度 16 步", fontsize=8.4, color=fs.MUTED,
            ha="left", va="center")

    fig.text(0.105, 0.955, "示意曲线：只表示误差随时域增长的方向与三条线的相对高低，"
             "刻度无数值意义", fontsize=8.3, color=fs.MUTED, ha="left", va="center")
    return fs.save(fig, "basics", "ch07", "rollout-error")


if __name__ == "__main__":
    fig_loop()
    fig_temperature()
    fig_imagine()
    fig_three_world_models()
    fig_rollout_error()
