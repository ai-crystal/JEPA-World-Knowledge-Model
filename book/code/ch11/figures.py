# -*- coding: utf-8 -*-
"""第 11 章插图：动作条件 JEPA 世界模型、潜空间规划示意。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_action_jepa():
    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- 第一行：当前帧 -> 编码器 -> z_t -> 预测器 -> ẑ_{t+1} ... ẑ_T
    fs.box(ax, 0.015, 0.705, 0.100, 0.160, "当前帧\n$x_t$", fc="#FFFFFF", ec=fs.INK,
           fs=9.0, lw=1.1)
    fs.arrow(ax, (0.117, 0.785), (0.152, 0.785), color=fs.INK, lw=1.1)

    fs.box(ax, 0.155, 0.705, 0.115, 0.160, "编码器 $E$\n冻结", fc="#F4F7FA", ec=fs.BLUE,
           fs=9.0, lw=1.2, tc=fs.BLUE)
    fs.arrow(ax, (0.272, 0.785), (0.300, 0.785), color=fs.INK, lw=1.1)

    fs.box(ax, 0.303, 0.705, 0.147, 0.160, "$z_t$\n16×16×1408", fc="#FFFFFF", ec=fs.BLUE,
           fs=7.6, lw=1.1)
    fs.arrow(ax, (0.452, 0.785), (0.487, 0.785), color=fs.INK, lw=1.1)

    fs.box(ax, 0.490, 0.640, 0.125, 0.290, "预测器 $P_\\phi$\n300M\n块因果注意力",
           fc="#FFF7F0", ec=fs.ACCENT, fs=8.6, lw=1.3, tc=fs.ACCENT)
    fs.arrow(ax, (0.617, 0.785), (0.652, 0.785), color=fs.INK, lw=1.1)

    fs.box(ax, 0.655, 0.705, 0.100, 0.160, "$\\hat z_{t+1}$\n预测表征", fc="#FFFFFF",
           ec=fs.ACCENT, fs=8.2, lw=1.1)
    fs.arrow(ax, (0.757, 0.785), (0.845, 0.785), color=fs.FAINT, lw=1.0, ls=(0, (3, 2.5)))
    fs.label(ax, 0.801, 0.672, "重复 $T$ 步", fs=7.4, color=fs.MUTED)

    fs.box(ax, 0.848, 0.705, 0.100, 0.160, "$\\hat z_T$\n$T$ 步之后", fc="#FFFFFF",
           ec=fs.ACCENT, fs=8.2, lw=1.1)

    # ---- 动作输入
    fs.box(ax, 0.395, 0.315, 0.270, 0.140, "动作 $a_t$ 与末端状态 $s_t$",
           fc="#FFFFFF", ec=fs.GREEN, fs=8.4, lw=1.2, tc=fs.GREEN)
    fs.label(ax, 0.395, 0.278, "各 7 维，逐帧差分", fs=7.4, color=fs.MUTED, ha="left")
    fs.arrow(ax, (0.530, 0.457), (0.548, 0.636), color=fs.GREEN, lw=1.3)

    # ---- 代价
    fs.box(ax, 0.795, 0.415, 0.155, 0.170, "代价 $C$\n$\\Vert \\hat z_T - z_g \\Vert_1$",
           fc="#FFFFFF", ec=fs.PURPLE, fs=8.0, lw=1.2, tc=fs.PURPLE)
    fs.arrow(ax, (0.898, 0.703), (0.885, 0.588), color=fs.PURPLE, lw=1.2)
    fs.arrow(ax, (0.752, 0.235), (0.820, 0.411), color=fs.PURPLE, lw=1.2)

    # ---- 第二行：目标分支
    fs.box(ax, 0.440, 0.095, 0.100, 0.140, "目标图\n$x_g$", fc="#FFFFFF", ec=fs.PURPLE,
           fs=9.0, lw=1.1)
    fs.arrow(ax, (0.542, 0.165), (0.563, 0.165), color=fs.INK, lw=1.1)
    fs.box(ax, 0.565, 0.095, 0.115, 0.140, "编码器 $E$", fc="#F4F7FA", ec=fs.BLUE,
           fs=8.6, lw=1.2, tc=fs.BLUE)
    fs.arrow(ax, (0.682, 0.165), (0.703, 0.165), color=fs.INK, lw=1.1)
    fs.box(ax, 0.705, 0.095, 0.090, 0.140, "$z_g$", fc="#FFFFFF", ec=fs.PURPLE,
           fs=9.0, lw=1.1)

    # ---- 左下角说明
    fs.label(ax, 0.030, 0.300, "目标图不参与训练", fs=7.8, color=fs.MUTED, ha="left")
    fs.label(ax, 0.030, 0.235, "只在规划时提供代价", fs=7.8, color=fs.MUTED, ha="left")

    return fs.save(fig, "ch11", "action-jepa")


def _bezier(p0, p1, p2, p3, n=260):
    t = np.linspace(0, 1, n).reshape(-1, 1)
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def fig_latent_planning():
    fig = plt.figure(figsize=(7.0, 3.3))
    axL = fig.add_axes([0.055, 0.135, 0.420, 0.735])
    axR = fig.add_axes([0.600, 0.215, 0.360, 0.600])

    z0 = np.array([1.20, 1.15])
    zg = np.array([8.40, 6.30])

    # 7 条候选轨迹，终点散在潜空间各处，弯曲程度不同
    cands = [
        ((8.45, 6.20), (1.9, 1.3), (1.9, 1.1)),   # 最优：终点几乎落在目标上
        ((6.10, 3.10), (1.4, 0.9), (1.5, 0.7)),
        ((4.30, 7.20), (1.1, 2.4), (1.2, 1.0)),
        ((9.30, 4.10), (2.3, 1.0), (1.4, 0.6)),
        ((4.90, 2.20), (1.7, 0.4), (1.3, 1.4)),
        ((9.70, 7.70), (2.0, 1.7), (1.6, 1.2)),
        ((3.00, 4.40), (1.2, 2.0), (1.1, 1.6)),
    ]
    curves, costs = [], []
    for (ex, ey), c1o, c2o in cands:
        e = np.array([ex, ey])
        c1 = z0 + np.array(c1o)
        c2 = e - np.array(c2o)
        p = _bezier(z0, c1, c2, e)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        curves.append(p)
        costs.append(np.linalg.norm(p[-1] - zg) + 0.09 * seg)

    order = list(np.argsort(costs))
    # 由浅到深：代价高的是浅灰，代价低的接近墨色
    ramp = ["#CBD5E1", "#B3BAC4", "#9CA3AF", "#78818F", "#616B7B", "#4B5563"]

    for rank, idx in enumerate(order[::-1]):     # 代价高的先画，最优的压在最上层
        best = (idx == order[0])
        c = fs.ACCENT if best else ramp[min(rank, len(ramp) - 1)]
        axL.plot(curves[idx][:, 0], curves[idx][:, 1], color=c,
                 lw=2.1 if best else 1.2, solid_capstyle="round",
                 zorder=30 if best else 3 + rank,
                 alpha=1.0 if best else 0.9)

    # 最优轨迹上的采样步，前几步实心表示真正执行
    best_p = curves[order[0]]
    ts = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    pts = best_p[np.round(np.array(ts) * (len(best_p) - 1)).astype(int)]
    for k, p in enumerate(pts[1:], start=1):
        if k <= 3:
            axL.scatter([p[0]], [p[1]], s=46, color=fs.ACCENT, edgecolor="white",
                        linewidth=1.0, zorder=32)
        else:
            axL.scatter([p[0]], [p[1]], s=26, facecolor="white", edgecolor=fs.ACCENT,
                        linewidth=0.9, zorder=32)

    axL.scatter([z0[0]], [z0[1]], s=72, color=fs.INK, edgecolor="white", linewidth=1.2, zorder=33)
    axL.scatter([zg[0]], [zg[1]], s=170, marker="*", color=fs.BLUE, edgecolor="white",
                linewidth=0.9, zorder=33)
    fs.label(axL, z0[0] + 0.35, z0[1] - 0.62, "当前状态 $z_0$", fs=8.4, color=fs.INK, ha="left")
    fs.label(axL, zg[0] + 0.45, zg[1] + 0.20, "目标 $z_g$", fs=8.4, color=fs.BLUE, ha="left")

    axL.annotate("实心点为实际执行的几步\n之后重新观察、重新规划",
                 xy=(pts[2][0] - 0.02, pts[2][1] - 0.06), xytext=(4.55, 0.35),
                 fontsize=7.4, color=fs.ACCENT, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                 shrinkA=2, shrinkB=3,
                                 connectionstyle="arc3,rad=-0.35"))

    axL.set_xlim(-0.2, 10.4); axL.set_ylim(-0.4, 8.0)
    axL.set_aspect("equal")
    axL.set_xticks([]); axL.set_yticks([])
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axL.spines[s].set_color("#9CA3AF")
    axL.set_xlabel("潜空间维度 1", fontsize=8.0, color=fs.MUTED, labelpad=2)
    axL.set_ylabel("潜空间维度 2", fontsize=8.0, color=fs.MUTED, labelpad=1)
    fs.label(axL, -0.05, 7.62, "二维示意，真实表征是上千维", fs=7.2, color=fs.FAINT, ha="left")
    axL.text(-0.03, 1.02, "(a)", transform=axL.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")

    # ---- 右图：按代价排序
    ranked = [costs[i] for i in order]
    ypos = np.arange(len(ranked))
    colors = [fs.ACCENT] + ["#CBD5E1"] * (len(ranked) - 1)
    axR.barh(ypos, ranked, height=0.62, color=colors, zorder=3)
    axR.set_yticks(ypos)
    axR.set_yticklabels([f"候选 {k + 1}" for k in range(len(ranked))], fontsize=8.2)
    axR.invert_yaxis()
    axR.set_xlim(0, max(ranked) * 1.42)
    axR.set_ylim(len(ranked) - 0.35, -1.45)
    axR.set_xticks([])
    axR.set_xlabel("代价（到目标表征的距离，越小越好）", fontsize=7.8, color=fs.MUTED, labelpad=3)
    axR.tick_params(axis="y", length=0, colors=fs.MUTED)
    for s in ("top", "right", "bottom"):
        axR.spines[s].set_visible(False)
    axR.spines["left"].set_color("#D1D5DB")
    axR.annotate("选中这条", xy=(ranked[0] * 0.72, -0.06), xytext=(ranked[0] * 1.02, -0.86),
                 fontsize=8.0, color=fs.ACCENT, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                 connectionstyle="arc3,rad=0.2"))
    axR.text(-0.03, 1.02, "(b)", transform=axR.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")

    return fs.save(fig, "ch11", "latent-planning")


if __name__ == "__main__":
    fig_action_jepa()
    fig_latent_planning()
