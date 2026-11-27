# -*- coding: utf-8 -*-
"""第 9 章插图：训练循环、显存分解、学习率调度、四种损失曲线。

数字来源：
- 参数 86M、12 层、768 维、12 头、patch 16：ViT 论文表 1（2010.11929）。
- 余弦调度与 15 轮 warmup（1e-4 → 1e-3 → 1e-6）：I-JEPA 附录 A.1（2301.08243）。
- warmup-constant-decay（12K / 228K / 12K，5.25e-4，batch 3072）：V-JEPA 2 表 9（2506.09985）。
- 显存各项由每参数字节数与张量形状手算，属粗估，正文与图注均标明。
- 四种损失曲线为示意形状，不含实测数据。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_training_loop():
    """图 9-1 一次迭代的三段路：前向、损失与反向、更新。"""
    fig = plt.figure(figsize=(7.0, 4.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 上排：一个批 → 前向 → 损失
    y1, h = 0.605, 0.165
    fs.box(ax, 0.030, y1, 0.150, h, "一个批\n$b\\times3\\times224\\times224$",
           fc="#FBF7F0", ec=fs.INK, fs=8.3, lw=1.1)
    fs.box(ax, 0.255, y1, 0.180, h, "前向\n$\\hat y = f_\\theta(x)$",
           fc="#F2F6FF", ec=fs.BLUE, fs=9.4, lw=1.4, tc=fs.BLUE)
    fs.box(ax, 0.495, y1, 0.150, h, "损失\n$L(\\hat y, y)$",
           fc="#FFF3EC", ec=fs.ACCENT, fs=9.4, lw=1.4, tc=fs.ACCENT)

    fs.arrow(ax, (0.182, 0.687), (0.253, 0.687), color=fs.INK, lw=1.1)
    fs.arrow(ax, (0.437, 0.687), (0.493, 0.687), color=fs.BLUE, lw=1.2)

    # 下排：反向 → 更新
    y2 = 0.235
    fs.box(ax, 0.495, y2, 0.150, h, "反向\n$\\partial L/\\partial\\theta$",
           fc="#FFF3EC", ec=fs.ACCENT, fs=9.4, lw=1.4, tc=fs.ACCENT)
    fs.box(ax, 0.715, y2, 0.180, h, "更新\n$\\theta\\leftarrow\\theta-\\eta\\,g$",
           fc="#EFFAF4", ec=fs.GREEN, fs=9.4, lw=1.4, tc=fs.GREEN)

    fs.arrow(ax, (0.570, 0.603), (0.570, 0.402), color=fs.ACCENT, lw=1.3)
    fs.label(ax, 0.585, 0.500, "算梯度", fs=8.4, color=fs.ACCENT, ha="left")
    fs.arrow(ax, (0.647, 0.318), (0.713, 0.318), color=fs.ACCENT, lw=1.3)

    # 回头路：更新 → 记一行日志 → 取下一批
    yb = 0.130
    ax.plot([0.805, 0.805], [0.233, yb], color=fs.GREEN, lw=1.2, zorder=2)
    ax.plot([0.805, 0.652], [yb, yb], color=fs.GREEN, lw=1.2, zorder=2)
    fs.box(ax, 0.452, yb - 0.050, 0.215, 0.100, "记日志\n损失、梯度范数、学习率",
           fc="#FFFFFF", ec=fs.GREEN, fs=7.4, lw=0.9, tc=fs.MUTED)
    ax.plot([0.450, 0.105], [yb, yb], color=fs.GREEN, lw=1.2, zorder=2)
    fs.arrow(ax, (0.105, yb), (0.105, 0.603), color=fs.GREEN, lw=1.2, ms=10)
    fs.label(ax, 0.360, 0.042, "下一步取新批，用新参数再算一遍", fs=8.4, color=fs.GREEN)

    # 标题与颜色说明
    fs.label(ax, 0.030, 0.960, "三段路，只有第三段改参数", fs=9.2, color=fs.INK, ha="left",
             weight="bold")
    legs = [(fs.BLUE, "前向：只读参数"), (fs.ACCENT, "损失与反向：只产生梯度"),
            (fs.GREEN, "更新：唯一改参数的一步")]
    x = 0.030
    for col, txt in legs:
        ax.plot([x, x + 0.028], [0.890, 0.890], color=col, lw=2.2, solid_capstyle="round",
                zorder=3)
        fs.label(ax, x + 0.036, 0.890, txt, fs=8.4, color=col, ha="left")
        x += 0.036 + 0.0175 * len(txt) + 0.032
    return fs.save(fig, "basics", "ch09", "training-loop")


def fig_memory():
    """图 9-2 显存分解：静态部分固定，激活随批大小和精度变。"""
    fig, ax = plt.subplots(figsize=(7.0, 3.75))
    fig.subplots_adjust(left=0.205, right=0.900, top=0.845, bottom=0.215)

    labels = ["全精度，批 32", "激活用 bf16，批 32", "bf16 加检查点，批 32"]
    comps = {
        "参数 4N": (np.array([0.344, 0.344, 0.344]), fs.GOLD),
        "梯度 4N": (np.array([0.344, 0.344, 0.344]), fs.MUTED),
        "Adam 一阶 4N": (np.array([0.344, 0.344, 0.344]), fs.BLUE),
        "Adam 二阶 4N": (np.array([0.344, 0.344, 0.344]), fs.PURPLE),
        "激活": (np.array([4.68, 2.34, 0.23]), fs.ACCENT),
    }
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for name, (vals, col) in comps.items():
        ax.barh(y, vals, left=left, height=0.46, color=col, edgecolor="white",
                linewidth=1.1, zorder=3, label=name)
        left += vals

    # 只标激活与合计
    ax.text(1.38 + 4.68 / 2, y[0], "4.7", fontsize=8.4, color="white", ha="center",
            va="center", zorder=5)
    ax.text(1.38 + 2.34 / 2, y[1], "2.3", fontsize=8.4, color="white", ha="center",
            va="center", zorder=5)
    ax.text(1.38 + 0.23 + 0.05, y[2] + 0.33, "0.23", fontsize=8.0, color=fs.ACCENT,
            ha="left", va="center", zorder=5)
    for yy, tot in zip(y, left):
        ax.text(tot + 0.10, yy, f"共 {tot:.1f} GB",
                fontsize=8.8, color=fs.INK, ha="left", va="center", zorder=5)

    # 静态部分的括注
    ax.annotate("", xy=(0.0, 2.40), xytext=(1.376, 2.40),
                arrowprops=dict(arrowstyle="<->", color=fs.MUTED, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text(0.69, 2.47, "静态 1.38 GB，三项相同", fontsize=8.0, color=fs.MUTED,
            ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.0, color=fs.INK)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(-0.55, 2.72)
    ax.set_xlabel("显存占用（GB，按 $10^9$ 字节计）", fontsize=9.2, color=fs.INK, labelpad=4)
    ax.tick_params(axis="x", labelsize=8.8)
    fs.clean(ax, spines=("bottom",))
    ax.grid(axis="x", color=fs.GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.010), ncol=5, fontsize=8.0,
              handlelength=1.2, columnspacing=1.0, handletextpad=0.5)

    fig.text(0.012, 0.020, "86M 参数；激活按每层存八个中间张量、197 个 token、768 维手算，属粗估",
             fontsize=7.8, color=fs.MUTED, ha="left", va="bottom")
    return fs.save(fig, "basics", "ch09", "memory-breakdown")


def fig_lr_schedule():
    """图 9-3 两种真实配方里的学习率曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.20))
    fig.subplots_adjust(left=0.090, right=0.985, top=0.845, bottom=0.175, wspace=0.235)

    # (a) I-JEPA：300 轮，前 15 轮从 1e-4 线性升到 1e-3，之后余弦降到 1e-6
    a = axes[0]
    ep = np.linspace(0, 300, 1600)
    lr = np.empty_like(ep)
    warm = ep < 15
    lr[warm] = 1e-4 + (1e-3 - 1e-4) * ep[warm] / 15
    t = (ep[~warm] - 15) / (300 - 15)
    lr[~warm] = 1e-6 + 0.5 * (1e-3 - 1e-6) * (1 + np.cos(np.pi * t))
    a.axvspan(0, 15, color="#FDF1E7", zorder=0)
    a.plot(ep, lr * 1e4, color=fs.ACCENT, lw=1.9, zorder=3)
    a.set_xlim(0, 300)
    a.set_ylim(0, 11.6)
    a.set_xticks([0, 50, 100, 150, 200, 250, 300])
    a.set_yticks([0, 2, 4, 6, 8, 10])
    a.set_xlabel("预训练轮数", fontsize=9.0, color=fs.INK, labelpad=4)
    a.set_ylabel("学习率（$10^{-4}$）", fontsize=9.0, color=fs.INK, labelpad=5)
    a.set_title("（a）15 轮升，余弦降（I-JEPA）", fontsize=9.2, color=fs.INK, pad=7)
    a.annotate("$10^{-4}\\rightarrow10^{-3}$", xy=(15, 10.0), xytext=(34, 10.55),
               fontsize=8.0, color=fs.ACCENT,
               arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8))
    a.annotate("尾部降到 $10^{-6}$", xy=(296, 0.06), xytext=(196, 2.55),
               fontsize=8.0, color=fs.MUTED,
               arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.8))
    fs.clean(a, spines=("left", "bottom"))
    a.grid(axis="y", color=fs.GRID, lw=0.6, zorder=0)

    # (b) V-JEPA 2：12K 升到 5.25e-4，恒定 228K，最后 12K 线性降到 1e-6
    b = axes[1]
    st = np.linspace(0, 252, 1600)
    lr2 = np.empty_like(st)
    w1 = st < 12
    lr2[w1] = 1e-4 + (5.25e-4 - 1e-4) * st[w1] / 12
    mid = (st >= 12) & (st <= 240)
    lr2[mid] = 5.25e-4
    tail = st > 240
    lr2[tail] = 5.25e-4 + (1e-6 - 5.25e-4) * (st[tail] - 240) / 12
    b.axvspan(0, 12, color="#EEF3FD", zorder=0)
    b.axvspan(240, 252, color="#EEF3FD", zorder=0)
    b.plot(st, lr2 * 1e4, color=fs.BLUE, lw=1.9, zorder=3)
    b.set_xlim(0, 252)
    b.set_ylim(0, 6.1)
    b.set_xticks([0, 50, 100, 150, 200, 250])
    b.set_yticks([0, 1, 2, 3, 4, 5])
    b.set_xlabel("预训练步数（千步）", fontsize=9.0, color=fs.INK, labelpad=4)
    b.set_ylabel("学习率（$10^{-4}$）", fontsize=9.0, color=fs.INK, labelpad=4)
    b.set_title("（b）升、恒定、降（V-JEPA 2）", fontsize=9.2, color=fs.INK, pad=7)
    b.text(22, 5.62, "恒定 228K 步，$5.25\\times10^{-4}$", fontsize=8.0, color=fs.BLUE,
           ha="left")
    b.annotate("尾段线性降到 $10^{-6}$", xy=(248, 0.5), xytext=(120, 1.90),
               fontsize=8.0, color=fs.MUTED,
               arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.8))
    fs.clean(b, spines=("left", "bottom"))
    b.grid(axis="y", color=fs.GRID, lw=0.6, zorder=0)
    return fs.save(fig, "basics", "ch09", "lr-schedule")


def fig_loss_curves():
    """图 9-4 四种曲线形状（示意）。"""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.30))
    fig.subplots_adjust(left=0.088, right=0.982, top=0.925, bottom=0.165,
                        wspace=0.20, hspace=0.42)
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 300)

    def jitter(y, s=0.010):
        return y * (1 + rng.normal(0, s, y.shape))

    tr_ok = 1.90 * np.exp(-3.1 * x) + 0.42
    va_ok = 1.90 * np.exp(-2.6 * x) + 0.50
    tr_of = 1.90 * np.exp(-3.6 * x) + 0.20
    va_of = 1.85 * np.exp(-3.0 * x) + 0.45 + 0.55 * np.clip(x - 0.50, 0, None) ** 1.6 * 3.2
    spike = 1 / (1 + np.exp(-(x - 0.545) / 0.018))
    tr_sp = 1.90 * np.exp(-2.8 * x) + 0.30 + 2.35 * spike
    va_sp = 1.85 * np.exp(-2.5 * x) + 0.38 + 2.55 * spike
    tr_pl = 2.05 - 0.16 * (1 - np.exp(-9 * x))
    va_pl = 2.16 - 0.10 * (1 - np.exp(-9 * x))

    panels = [
        ("（a）健康", tr_ok, va_ok, "两条都在降，间隙稳定", fs.GREEN),
        ("（b）过拟合", tr_of, va_of, "训练继续降，验证先平后升", fs.ACCENT),
        ("（c）发散", tr_sp, va_sp, "损失突然向上翻", fs.PURPLE),
        ("（d）平台", tr_pl, va_pl, "两条都平，离下限很远", fs.GOLD),
    ]
    for ax, (title, ytr, yva, note, col) in zip(axes.ravel(), panels):
        ax.plot(x, jitter(ytr), color=col, lw=1.5, zorder=3, label="训练")
        ax.plot(x, jitter(yva, 0.005), color=fs.INK, lw=1.3, ls=(0, (4, 2.5)), zorder=4,
                label="验证")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 3.35)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "一半", "结束"], fontsize=8.2)
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.2, color=fs.INK, pad=5, loc="left")
        ax.text(0.030, 3.08, note, fontsize=7.8, color=col, ha="left", va="top")
        fs.clean(ax, spines=("left", "bottom"))
        ax.grid(axis="y", color=fs.GRID, lw=0.6, zorder=0)

    axes[0, 1].annotate("", xy=(0.70, va_of[210]), xytext=(0.44, 1.95),
                        arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.1,
                                        shrinkA=1, shrinkB=3))
    axes[0, 1].text(0.60, 2.32, "开始记住训练集", fontsize=7.8, color=fs.ACCENT, ha="left")
    axes[1, 0].annotate("", xy=(0.60, 2.20), xytext=(0.60, 1.02),
                        arrowprops=dict(arrowstyle="-|>", color=fs.PURPLE, lw=1.1,
                                        shrinkA=2, shrinkB=3))
    axes[1, 0].text(0.63, 1.55, "梯度尖峰", fontsize=7.8, color=fs.PURPLE, ha="left")

    axes[0, 0].set_ylabel("损失", fontsize=8.8, color=fs.INK, labelpad=3)
    axes[1, 0].set_ylabel("损失", fontsize=8.8, color=fs.INK, labelpad=3)
    for ax in axes[1]:
        ax.set_xlabel("训练进度", fontsize=8.8, color=fs.INK, labelpad=3)
    axes[0, 0].legend(loc="upper right", fontsize=7.8, handlelength=1.6,
                      bbox_to_anchor=(1.0, 1.03), ncol=2, columnspacing=1.0)
    fig.text(0.012, 0.018, "四条曲线为示意形状，不含实测数据", fontsize=7.8,
             color=fs.MUTED, ha="left", va="bottom")
    return fs.save(fig, "basics", "ch09", "loss-curves")


if __name__ == "__main__":
    fig_training_loop()
    fig_memory()
    fig_lr_schedule()
    fig_loss_curves()
