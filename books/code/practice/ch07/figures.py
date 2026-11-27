# -*- coding: utf-8 -*-
"""第 7 章插图：RSSM 结构、想象轨迹、规划延迟对比、损失构成、延迟分解。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_rssm():
    """图 7-1 RSSM 的五个部件与两条通路。"""
    fig = plt.figure(figsize=(7.0, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- 左侧：上一步留下来的三个量
    fs.box(ax, 0.018, 0.775, 0.090, 0.115, "$h_{t-1}$", fc="#FFFFFF", ec=fs.INK, fs=9.2, lw=1.1)
    fs.box(ax, 0.018, 0.590, 0.090, 0.115, "$z_{t-1}$", fc="#FFFFFF", ec=fs.INK, fs=9.2, lw=1.1)
    fs.box(ax, 0.018, 0.405, 0.090, 0.115, "$a_{t-1}$", fc="#FFFFFF", ec=fs.INK, fs=9.2, lw=1.1)

    # ---- 序列模型
    fs.box(ax, 0.140, 0.385, 0.110, 0.515, "序列模型\n$f_\\phi$", fc="#F4F7FA", ec=fs.BLUE,
           fs=9.0, lw=1.3, tc=fs.BLUE)
    for y in (0.832, 0.647, 0.462):
        fs.arrow(ax, (0.110, y), (0.137, y), color=fs.INK, lw=1.0)

    # ---- h_t
    fs.box(ax, 0.290, 0.635, 0.080, 0.130, "$h_t$", fc="#EEF3FF", ec=fs.BLUE, fs=9.4, lw=1.3)
    fs.arrow(ax, (0.252, 0.700), (0.287, 0.700), color=fs.BLUE, lw=1.2)

    # ---- 先验通路：动力学预测器 -> ẑ_t
    fs.box(ax, 0.412, 0.775, 0.132, 0.125, "动力学预测器\n$p_\\phi$", fc="#FFF7F0",
           ec=fs.ACCENT, fs=8.6, lw=1.3, tc=fs.ACCENT)
    fs.arrow(ax, (0.372, 0.740), (0.409, 0.808), color=fs.ACCENT, lw=1.2)
    fs.box(ax, 0.585, 0.775, 0.080, 0.125, "$\\hat z_t$", fc="#FFFFFF", ec=fs.ACCENT,
           fs=9.2, lw=1.2)
    fs.arrow(ax, (0.546, 0.837), (0.582, 0.837), color=fs.ACCENT, lw=1.2)

    # ---- 后验通路：观测 -> 编码器 -> z_t
    fs.box(ax, 0.290, 0.435, 0.080, 0.125, "观测\n$x_t$", fc="#FFFFFF", ec=fs.INK, fs=8.4, lw=1.0)
    fs.box(ax, 0.412, 0.435, 0.132, 0.125, "编码器\n$q_\\phi$", fc="#F0FBF6",
           ec=fs.GREEN, fs=8.6, lw=1.3, tc=fs.GREEN)
    fs.arrow(ax, (0.372, 0.497), (0.409, 0.497), color=fs.INK, lw=1.0)
    fs.arrow(ax, (0.330, 0.632), (0.430, 0.563), color=fs.GREEN, lw=1.1)
    fs.box(ax, 0.585, 0.435, 0.080, 0.125, "$z_t$", fc="#EEF9F2", ec=fs.GREEN, fs=9.4, lw=1.3)
    fs.arrow(ax, (0.546, 0.497), (0.582, 0.497), color=fs.GREEN, lw=1.2)

    # ---- KL：先验与后验之间
    ax.annotate("", xy=(0.625, 0.570), xytext=(0.625, 0.768),
                arrowprops=dict(arrowstyle="<|-|>", color=fs.PURPLE, lw=1.2,
                                shrinkA=1, shrinkB=1))
    fs.label(ax, 0.618, 0.668, "KL", fs=8.6, color=fs.PURPLE, ha="right", weight="bold")

    # ---- h_t 到三个头的条件输入
    fs.arrow(ax, (0.372, 0.700), (0.694, 0.700), color=fs.MUTED, lw=0.9,
             ls=(0, (4, 2.6)), ms=9)
    fs.label(ax, 0.512, 0.722, "条件输入", fs=7.6, color=fs.MUTED)

    # ---- 汇合点与总线
    ax.add_patch(plt.Circle((0.700, 0.4975), 0.0075, facecolor="white", edgecolor=fs.MUTED,
                            lw=0.9, zorder=8))
    ax.plot([0.700, 0.700], [0.3525, 0.7525], color=fs.MUTED, lw=0.9, zorder=2)
    ax.plot([0.665, 0.6925], [0.4975, 0.4975], color=fs.MUTED, lw=0.9, zorder=2)

    heads = [(0.7525, "解码器　$\\hat x_t$", fs.INK, "#FFFFFF"),
             (0.5525, "奖励头　$\\hat r_t$", fs.INK, "#FFFFFF"),
             (0.3525, "终止头　$\\hat c_t$", fs.INK, "#FFFFFF")]
    for yc, txt, col, fc in heads:
        fs.arrow(ax, (0.703, yc), (0.749, yc), color=fs.MUTED, lw=0.9, ms=8)
        fs.box(ax, 0.752, yc - 0.0525, 0.192, 0.105, txt, fc=fc, ec=col, fs=8.8, lw=1.2)

    # ---- 回流：h_t 与 z_t 进入下一步
    ax.plot([0.625, 0.625], [0.432, 0.215], color=fs.BLUE, lw=0.9,
            ls=(0, (4, 2.6)), zorder=2)
    ax.plot([0.195, 0.625], [0.215, 0.215], color=fs.BLUE, lw=0.9,
            ls=(0, (4, 2.6)), zorder=2)
    fs.arrow(ax, (0.195, 0.215), (0.195, 0.382), color=fs.BLUE, lw=0.9,
             ls=(0, (4, 2.6)), ms=9)
    fs.label(ax, 0.400, 0.243, "下一步的两个量回到序列模型", fs=7.6, color=fs.BLUE)

    # ---- 底部注记
    fs.label(ax, 0.018, 0.135, "想象时没有观测，$z_t$ 改由先验 $\\hat z_t$ 采样，编码器整条支路省掉",
             fs=7.8, color=fs.MUTED, ha="left")
    fs.label(ax, 0.018, 0.062, "这条 KL 被算两次，两边各停一次梯度，各自压到 1 nat 以下就不再计损失",
             fs=7.8, color=fs.MUTED, ha="left")

    return fs.save(fig, "practice", "ch07", "rssm")


def fig_imagination():
    """图 7-2 真实轨迹与想象轨迹。"""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.135, top=0.855, wspace=0.16)

    for ax in axes:
        ax.set_xlim(0.2, 8.6)
        ax.set_ylim(0.4, 8.9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("潜空间的第一维（示意）", fontsize=8.2, color=fs.MUTED)
        ax.set_ylabel("第二维", fontsize=8.2, color=fs.MUTED, labelpad=1)
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
    for i, (x, y) in enumerate(real):
        a.text(x, y - 0.60, f"$s_{i + 1}$", fontsize=8.8, color=fs.BLUE,
               ha="center", va="top", zorder=5)
    a.set_title("（a）回放缓冲里的一段真实片段", fontsize=9.6, color=fs.INK, pad=8)
    a.text(0.30, 8.05, "观测逐帧过编码器", fontsize=8.4, color=fs.MUTED, ha="left")

    # (b) 想象轨迹：从同一状态出发，策略采样动作，动力学推进
    b = axes[1]
    start = np.array([1.6, 4.4])
    paths = [
        (fs.ACCENT, [[2.7, 5.2], [3.9, 5.7], [5.1, 6.3], [6.3, 6.7], [7.4, 7.4]]),
        (fs.PURPLE, [[2.8, 4.0], [4.0, 3.4], [5.2, 3.0], [6.4, 2.5], [7.5, 2.4]]),
        (fs.GREEN, [[2.6, 4.5], [3.8, 4.6], [5.0, 4.1], [6.2, 4.7], [7.3, 4.9]]),
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
    b.text(start[0] - 0.12, start[1] + 0.55, "$s_t$", fontsize=9.4, color=fs.INK, ha="left")
    b.set_title("（b）从同一状态出发的想象轨迹", fontsize=9.6, color=fs.INK, pad=8)
    b.text(0.30, 8.05, "同一状态、不同动作序列", fontsize=8.4, color=fs.MUTED, ha="left")
    b.annotate("", xy=(7.5, 1.60), xytext=(2.6, 1.60),
               arrowprops=dict(arrowstyle="<->", color=fs.FAINT, lw=0.9))
    b.text(5.05, 1.18, "在动力学里推进 $T=16$ 步", fontsize=8.4, color=fs.MUTED, ha="center")

    return fs.save(fig, "practice", "ch07", "imagination")


def fig_latency():
    """图 7-3 单个动作与整条抓放轨迹的规划耗时。"""
    per_action = [("V-JEPA 2-AC", 16.0, fs.ACCENT), ("Cosmos", 240.0, fs.FAINT)]
    steps = 18                       # 4 + 10 + 4，三个子目标各一段
    per_traj = [(n, s * steps / 60.0, c) for n, s, c in per_action]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.85))
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.205, top=0.775, wspace=0.30)

    for ax, data, unit, note in (
            (axes[0], per_action, "秒", "单个规划动作，单块 4090\n800 条采样、10 轮迭代"),
            (axes[1], per_traj, "分钟", "整条抓放轨迹\n%d 个规划步" % steps)):
        xs = np.arange(len(data))
        vals = [d[1] for d in data]
        ax.bar(xs, vals, width=0.44, color=[d[2] for d in data], zorder=3)
        for x, val in zip(xs, vals):
            txt = f"{val:.0f} {unit}" if val >= 10 else f"{val:.1f} {unit}"
            ax.text(x, val * 1.045, txt, fontsize=9.0, color=fs.INK, ha="center", va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels([d[0] for d in data], fontsize=8.8, color=fs.INK)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, pad=5)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#D1D5DB")
        ax.set_xlabel(note, fontsize=7.6, color=fs.MUTED, labelpad=6, linespacing=1.5)

    axes[0].text(0.0, 1.075, "（a）单个动作", transform=axes[0].transAxes, fontsize=9.4,
                 color=fs.MUTED, weight="bold", ha="left", va="bottom")
    axes[1].text(0.0, 1.075, "（b）整条轨迹", transform=axes[1].transAxes, fontsize=9.4,
                 color=fs.MUTED, weight="bold", ha="left", va="bottom")
    return fs.save(fig, "practice", "ch07", "planning-latency")


def fig_losses():
    """图 7-4 DreamerV3 的五条损失与 free bits 的开关作用。"""
    fig = plt.figure(figsize=(7.0, 3.55))
    ax = fig.add_axes([0.005, 0.015, 0.700, 0.965])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- 上半：世界模型的三条损失
    fs.label(ax, 0.012, 0.968, "世界模型 $\\phi$：整段真实片段上端到端算", fs=8.8,
             color=fs.INK, ha="left", weight="bold")

    rows = [
        (0.870, "预测损失", "$\\beta_{\\mathrm{pred}}=1$", "解码器 · 奖励头 · 终止头", fs.INK, "#FFFFFF", None),
        (0.720, "动力学损失", "$\\beta_{\\mathrm{dyn}}=1$", "序列模型 $f_\\phi$", fs.BLUE, "#F4F7FA",
         "$\\mathrm{sg}(q_\\phi)$"),
        (0.570, "表征损失", "$\\beta_{\\mathrm{rep}}=0.1$", "编码器 $q_\\phi$", fs.GREEN, "#F0FBF6",
         "$\\mathrm{sg}(p_\\phi)$"),
    ]
    for yc, name, coef, mod, col, fc, sg in rows:
        fs.box(ax, 0.012, yc - 0.0575, 0.245, 0.115, "", fc=fc, ec=col, lw=1.2)
        fs.label(ax, 0.134, yc + 0.026, name, fs=8.6, color=col, weight="bold")
        fs.label(ax, 0.134, yc - 0.028, coef, fs=8.2, color=col)
        fs.arrow(ax, (0.262, yc), (0.352, yc), color=col, lw=1.1, ms=9)
        if sg:
            fs.label(ax, 0.307, yc + 0.048, sg, fs=6.8, color=fs.MUTED)
        fs.box(ax, 0.357, yc - 0.0575, 0.330, 0.115, mod, fc="#FFFFFF", ec=col, fs=7.8, lw=1.0)

    # ---- 分隔线
    ax.plot([0.012, 0.690], [0.495, 0.495], color=fs.GRID, lw=1.0, ls=(0, (3, 2.4)))
    fs.label(ax, 0.012, 0.462, "actor 与 critic：在 16 步想象轨迹上算，不碰真实观测", fs=8.8,
             color=fs.INK, ha="left", weight="bold")

    rows2 = [
        (0.375, "价值损失", "$\\beta_{\\mathrm{val}}=1$", "critic $v_\\psi$，symexp twohot", fs.ACCENT),
        (0.225, "策略损失", "$\\beta_{\\mathrm{pol}}=1$", "actor $\\pi_\\theta$，熵 $\\eta=3\\times10^{-4}$", fs.PURPLE),
    ]
    for yc, name, coef, mod, col in rows2:
        fs.box(ax, 0.012, yc - 0.0575, 0.245, 0.115, "", fc="#FFFFFF", ec=col, lw=1.2)
        fs.label(ax, 0.134, yc + 0.026, name, fs=8.6, color=col, weight="bold")
        fs.label(ax, 0.134, yc - 0.028, coef, fs=8.2, color=col)
        fs.arrow(ax, (0.262, yc), (0.352, yc), color=col, lw=1.1, ms=9)
        fs.box(ax, 0.357, yc - 0.0575, 0.330, 0.115, mod, fc="#FFFFFF", ec=col, fs=7.8, lw=1.0)

    # ---- 底部两条注记
    fs.label(ax, 0.012, 0.100, "free bits：两条 KL 压到 1 nat（约 1.44 bit）以下就不再计损失",
             fs=7.6, color=fs.MUTED, ha="left")
    fs.label(ax, 0.012, 0.038, "1% 均匀混合：分组高斯的每一类概率都压不到零，KL 不会尖峰",
             fs=7.6, color=fs.MUTED, ha="left")

    # ---- 右：free bits 的权重曲线
    axb = fig.add_axes([0.775, 0.255, 0.215, 0.560])
    kl = np.linspace(0.0, 3.0, 601)
    w_dyn = np.where(kl > 1.0, 1.0, 0.0)
    w_rep = np.where(kl > 1.0, 0.1, 0.0)
    axb.axvspan(0.0, 1.0, color="#F3F4F6", zorder=0)
    axb.step(kl, w_dyn, where="post", color=fs.BLUE, lw=1.6, zorder=3)
    axb.step(kl, w_rep, where="post", color=fs.GREEN, lw=1.6, zorder=3)
    axb.set_xlim(0, 3); axb.set_ylim(-0.05, 1.18)
    axb.set_yticks([0, 0.1, 1]); axb.set_yticklabels(["0", "0.1", "1"], fontsize=7.6)
    axb.set_xticks([0, 1, 2, 3]); axb.tick_params(labelsize=7.6, length=2.5)
    axb.set_xlabel("KL（nat）", fontsize=8.0, labelpad=2)
    axb.set_ylabel("该损失的有效权重", fontsize=8.0, labelpad=2)
    fs.clean(axb)
    axb.text(0.5, 0.62, "梯度为 0", fontsize=7.2, color=fs.FAINT, ha="center")
    axb.annotate("", xy=(1.0, 1.02), xytext=(1.62, 1.10),
                 arrowprops=dict(arrowstyle="-|>", color=fs.MUTED, lw=0.8, shrinkA=1, shrinkB=2))
    axb.text(1.66, 1.10, "$\\beta_{\\mathrm{dyn}}=1$", fontsize=7.4, color=fs.BLUE, ha="left", va="center")
    axb.text(1.18, 0.19, "$\\beta_{\\mathrm{rep}}=0.1$", fontsize=7.4, color=fs.GREEN, ha="left")
    axb.text(0.0, 1.10, "（b）", fontsize=9.0, color=fs.MUTED, weight="bold",
             transform=axb.transAxes)
    ax.text(0.0, 1.005, "（a）", fontsize=9.0, color=fs.MUTED, weight="bold",
            transform=ax.transAxes, va="bottom")

    return fs.save(fig, "practice", "ch07", "dreamer-losses")


def _latency_model(n_samples=800, n_iter=10, horizon=1, ms_forward=1.80,
                  ms_overhead=0.20, s_encode=0.03, s_book=0.01):
    """按“每个候选一次前向”分摊一次规划动作的耗时，总量对齐论文报告的 16 秒。"""
    n_fwd = n_samples * n_iter * horizon
    s_fwd = n_fwd * ms_forward / 1000.0
    s_oh = n_fwd * ms_overhead / 1000.0
    return dict(n_fwd=n_fwd, forward=s_fwd, overhead=s_oh,
                encode=s_encode * horizon, book=s_book * horizon,
                total=s_fwd + s_oh + s_encode * horizon + s_book * horizon)


def fig_latency_breakdown():
    """图 7-5 一次规划动作的 16 秒分解与随采样数的增长。"""
    m = _latency_model()
    items = [("预测器前向", m["forward"], fs.ACCENT),
             ("框架与同步开销", m["overhead"], fs.FAINT),
             ("编码器前向", m["encode"], fs.BLUE),
             ("采样 / 排序 / 更新", m["book"], fs.GREEN)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95))
    fig.subplots_adjust(left=0.175, right=0.980, bottom=0.205, top=0.790, wspace=0.42)

    # (a) 横向条形：谁吃掉了这 16 秒
    a = axes[0]
    ys = np.arange(len(items))[::-1]
    for y, (name, val, col) in zip(ys, items):
        a.barh(y, val, height=0.56, color=col, zorder=3)
        pct = 100.0 * val / m["total"]
        if pct < 1.0:
            a.plot([val, val], [y - 0.28, y + 0.28], color=col, lw=3.0, solid_capstyle="butt", zorder=4)
        txt = f"{val:.2f} 秒"
        a.text(val + 0.42, y, txt if pct >= 1 else f"{txt}（{pct:.1f}%）",
               fontsize=7.8, color=fs.INK, va="center", ha="left")
    a.set_yticks(ys)
    a.set_yticklabels([it[0] for it in items], fontsize=8.2, color=fs.INK)
    a.set_xlim(0, 20.5); a.set_xticks([0, 5, 10, 15])
    a.set_xlabel("一次规划动作的累计耗时（秒），共 %d 次前向" % m["n_fwd"],
                 fontsize=7.8, labelpad=3)
    a.tick_params(axis="y", length=0, pad=4); a.tick_params(axis="x", labelsize=7.8, length=2.5)
    fs.clean(a)
    a.text(0.0, 1.10, "（a）16 秒的去向", transform=a.transAxes, fontsize=9.2,
           color=fs.MUTED, weight="bold", ha="left", va="bottom")

    # (b) 延迟随采样条数与时域的线性增长
    b = axes[1]
    ns = np.array([80, 200, 400, 800, 1600])
    for hz, col, lab in ((1, fs.ACCENT, "时域 1"), (3, fs.BLUE, "时域 3")):
        ts = np.array([_latency_model(n_samples=n, horizon=hz)["total"] for n in ns])
        b.plot(ns, ts, color=col, lw=1.5, marker="o", ms=3.6, zorder=3, label=lab)
    t800 = _latency_model()["total"]
    b.scatter([800], [t800], s=54, facecolor="white", edgecolor=fs.INK, lw=1.1, zorder=5)
    b.annotate("800 条、16 秒", xy=(800, t800), xytext=(880, 4.2), fontsize=7.8,
               color=fs.INK, ha="left",
               arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.8))
    b.set_xscale("log")
    b.set_xticks(ns); b.set_xticklabels([str(n) for n in ns], fontsize=7.6)
    b.set_xlim(60, 2100); b.set_ylim(0, 105)
    b.set_xlabel("每轮采样的候选条数", fontsize=8.0, labelpad=3)
    b.set_ylabel("一次动作的耗时（秒）", fontsize=8.0, labelpad=3)
    b.tick_params(labelsize=7.6, length=2.5)
    b.legend(loc="upper left", fontsize=7.6, handlelength=1.4, borderpad=0.2)
    fs.clean(b, grid=True)
    b.text(0.0, 1.10, "（b）延迟随预算线性长", transform=b.transAxes, fontsize=9.2,
           color=fs.MUTED, weight="bold", ha="left", va="bottom")

    print("n_fwd=%d, forward=%.2fs, overhead=%.2fs, encode=%.3fs, book=%.3fs, total=%.2fs"
          % (m["n_fwd"], m["forward"], m["overhead"], m["encode"], m["book"], m["total"]))
    print("800 samples x 10 iter x 1 step: %.1f%% forward, %.1f%% overhead"
          % (100 * m["forward"] / m["total"], 100 * m["overhead"] / m["total"]))
    return fs.save(fig, "practice", "ch07", "latency-breakdown")


if __name__ == "__main__":
    fig_rssm()
    fig_imagination()
    fig_latency()
    fig_losses()
    fig_latency_breakdown()
    print("per-action 16 s vs 240 s; trajectory 18 steps -> %.1f min vs %.1f min"
          % (16.0 * 18 / 60, 240.0 * 18 / 60))
