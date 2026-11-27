# -*- coding: utf-8 -*-
"""第 8 章插图：动作条件 JEPA 世界模型、潜空间规划示意。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_action_jepa():
    fig = plt.figure(figsize=(7.0, 3.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- 第一行：当前帧 -> 编码器 -> z_t -> 预测器 -> ẑ_{t+1} -> ẑ_T
    fs.box(ax, 0.020, 0.660, 0.090, 0.150, "当前帧\n$x_t$", fc="#FFFFFF", ec=fs.INK,
           fs=9.0, lw=1.1)
    fs.arrow(ax, (0.112, 0.735), (0.145, 0.735), color=fs.INK, lw=1.1)

    fs.box(ax, 0.147, 0.660, 0.110, 0.150, "编码器 $E$\n冻结", fc="#F4F7FA", ec=fs.BLUE,
           fs=9.0, lw=1.2, tc=fs.BLUE)
    fs.arrow(ax, (0.259, 0.735), (0.292, 0.735), color=fs.INK, lw=1.1)

    fs.box(ax, 0.294, 0.660, 0.140, 0.150, "$z_t$\n16×16×1408", fc="#FFFFFF", ec=fs.BLUE,
           fs=7.6, lw=1.1)
    fs.arrow(ax, (0.436, 0.735), (0.470, 0.735), color=fs.INK, lw=1.1)

    fs.box(ax, 0.472, 0.590, 0.120, 0.270, "预测器 $P_\\phi$\n300M\n块因果注意力",
           fc="#FFF7F0", ec=fs.ACCENT, fs=8.4, lw=1.3, tc=fs.ACCENT)
    fs.arrow(ax, (0.594, 0.735), (0.628, 0.735), color=fs.INK, lw=1.1)

    fs.box(ax, 0.630, 0.660, 0.098, 0.150, "$\\hat z_{t+1}$", fc="#FFFFFF",
           ec=fs.ACCENT, fs=8.4, lw=1.1)
    fs.arrow(ax, (0.730, 0.735), (0.790, 0.735), color=fs.FAINT, lw=1.0, ls=(0, (3, 2.5)))
    fs.label(ax, 0.760, 0.618, "重复 $T$ 步", fs=7.4, color=fs.MUTED)

    fs.box(ax, 0.792, 0.660, 0.098, 0.150, "$\\hat z_T$\n$T$ 步之后", fc="#FFFFFF",
           ec=fs.ACCENT, fs=8.0, lw=1.1)

    # ---- 自回归回路：输出接回输入
    fs.arrow(ax, (0.700, 0.815), (0.545, 0.865), color=fs.ACCENT, lw=1.15,
             ls=(0, (3, 2)), rad=0.42)
    fs.label(ax, 0.625, 0.945, "输出接回输入", fs=7.6, color=fs.ACCENT)

    # ---- 动作输入
    fs.box(ax, 0.355, 0.290, 0.270, 0.135, "动作 $a_t$ 与末端状态 $s_t$",
           fc="#FFFFFF", ec=fs.GREEN, fs=8.4, lw=1.2, tc=fs.GREEN)
    fs.label(ax, 0.355, 0.248, "各 7 维，动作是相邻两帧状态之差", fs=7.2, color=fs.MUTED,
             ha="left")
    fs.arrow(ax, (0.505, 0.428), (0.522, 0.587), color=fs.GREEN, lw=1.3)

    # ---- 代价
    fs.box(ax, 0.795, 0.365, 0.170, 0.145, "代价 $C$\n$\\Vert \\hat z_T - z_g \\Vert_1$",
           fc="#FFFFFF", ec=fs.PURPLE, fs=8.0, lw=1.2, tc=fs.PURPLE)
    fs.arrow(ax, (0.856, 0.656), (0.868, 0.514), color=fs.PURPLE, lw=1.2)

    # ---- 第二行：目标分支
    fs.box(ax, 0.440, 0.075, 0.098, 0.135, "目标图\n$x_g$", fc="#FFFFFF", ec=fs.PURPLE,
           fs=9.0, lw=1.1)
    fs.arrow(ax, (0.540, 0.142), (0.573, 0.142), color=fs.INK, lw=1.1)
    fs.box(ax, 0.575, 0.075, 0.108, 0.135, "编码器 $E$", fc="#F4F7FA", ec=fs.BLUE,
           fs=8.6, lw=1.2, tc=fs.BLUE)
    fs.arrow(ax, (0.685, 0.142), (0.716, 0.142), color=fs.INK, lw=1.1)
    fs.box(ax, 0.718, 0.075, 0.080, 0.135, "$z_g$", fc="#FFFFFF", ec=fs.PURPLE,
           fs=9.0, lw=1.1)
    fs.arrow(ax, (0.762, 0.214), (0.824, 0.361), color=fs.PURPLE, lw=1.2)

    # ---- 左下角说明
    fs.label(ax, 0.026, 0.182, "目标图不参与训练", fs=7.8, color=fs.MUTED, ha="left")
    fs.label(ax, 0.026, 0.116, "只在规划时提供代价", fs=7.8, color=fs.MUTED, ha="left")
    fs.arrow(ax, (0.212, 0.142), (0.434, 0.142), color=fs.FAINT, lw=0.9, ls=(0, (3, 2.5)))

    return fs.save(fig, "basics", "ch08", "action-jepa")


def _bezier(p0, p1, p2, p3, n=260):
    t = np.linspace(0, 1, n).reshape(-1, 1)
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def fig_latent_planning():
    fig = plt.figure(figsize=(7.0, 3.4))
    axL = fig.add_axes([0.070, 0.140, 0.400, 0.720])
    axR = fig.add_axes([0.610, 0.210, 0.345, 0.590])

    z0 = np.array([1.10, 1.00])
    zg = np.array([8.55, 6.35])

    # 7 条候选轨迹，终点散在潜空间各处，弯曲程度不同
    cands = [
        ((8.50, 6.25), (1.7, 1.1), (1.6, 1.0)),   # 1：终点几乎落在目标上
        ((6.15, 2.85), (1.5, 0.6), (1.4, 0.6)),
        ((4.15, 6.70), (0.9, 2.2), (1.2, 0.9)),
        ((8.85, 3.85), (2.2, 0.9), (1.3, 0.5)),
        ((4.70, 1.85), (1.6, 0.3), (1.2, 1.2)),
        ((9.95, 7.30), (1.9, 1.6), (1.5, 1.1)),
        ((2.85, 4.20), (1.0, 1.9), (1.0, 1.5)),
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

    # 编号位置手工指定，避免压到曲线和目标标记
    tag_pos = [(6.90, 5.78), (6.30, 3.10), (4.32, 7.00), (9.05, 4.02),
               (4.90, 1.62), (10.12, 7.42), (3.12, 4.52)]

    order = list(np.argsort(costs))
    ramp = ["#CBD5E1", "#B3BAC4", "#9CA3AF", "#808A98", "#616B7B", "#4B5563"]

    for rank, idx in enumerate(order[::-1]):     # 代价高的先画，最优的压在最上层
        best = (idx == order[0])
        c = fs.ACCENT if best else ramp[min(rank, len(ramp) - 1)]
        axL.plot(curves[idx][:, 0], curves[idx][:, 1], color=c,
                 lw=2.1 if best else 1.2, solid_capstyle="round",
                 zorder=30 if best else 3 + rank,
                 alpha=1.0 if best else 0.92)
        lx, ly = tag_pos[idx]
        axL.text(lx, ly, str(idx + 1), fontsize=7.2,
                 color=fs.ACCENT if best else fs.MUTED, ha="left", va="center", zorder=31)

    # 最优轨迹上的采样步，前几步实心表示真正执行
    best_p = curves[order[0]]
    ts = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    pts = best_p[np.round(np.array(ts) * (len(best_p) - 1)).astype(int)]
    for k, p in enumerate(pts[1:], start=1):
        if k <= 3:
            axL.scatter([p[0]], [p[1]], s=44, color=fs.ACCENT, edgecolor="white",
                        linewidth=1.0, zorder=32)
        else:
            axL.scatter([p[0]], [p[1]], s=24, facecolor="white", edgecolor=fs.ACCENT,
                        linewidth=0.9, zorder=32)

    axL.scatter([z0[0]], [z0[1]], s=70, color=fs.INK, edgecolor="white", linewidth=1.2, zorder=33)
    axL.scatter([zg[0]], [zg[1]], s=165, marker="*", color=fs.BLUE, edgecolor="white",
                linewidth=0.9, zorder=33)
    fs.label(axL, z0[0] - 0.15, z0[1] - 0.60, "当前状态 $z_0$", fs=8.4, color=fs.INK, ha="left")
    fs.label(axL, zg[0] - 1.55, zg[1] + 0.55, "目标 $z_g$", fs=8.4, color=fs.BLUE, ha="left")

    axL.annotate("实心点为实际执行的几步\n之后重新观察、重新规划",
                 xy=(pts[2][0] - 0.02, pts[2][1] - 0.10), xytext=(3.55, 0.10),
                 fontsize=7.4, color=fs.ACCENT, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                 shrinkA=2, shrinkB=3,
                                 connectionstyle="arc3,rad=-0.30"))

    axL.set_xlim(-0.4, 10.9); axL.set_ylim(-0.6, 8.3)
    axL.set_aspect("equal")
    axL.set_xticks([]); axL.set_yticks([])
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axL.spines[s].set_color("#9CA3AF")
    axL.set_xlabel("潜空间维度 1", fontsize=8.0, color=fs.MUTED, labelpad=2)
    axL.set_ylabel("潜空间维度 2", fontsize=8.0, color=fs.MUTED, labelpad=1)
    axL.text(0.035, 0.985, "二维示意，真实表征是上千维", transform=axL.transAxes,
             fontsize=7.2, color=fs.FAINT, ha="left", va="top")
    axL.text(-0.055, 1.03, "(a)", transform=axL.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")

    # ---- 右图：按代价排序
    ranked = [costs[i] for i in order]
    ypos = np.arange(len(ranked))
    colors = [fs.ACCENT] + ["#CBD5E1"] * (len(ranked) - 1)
    axR.barh(ypos, ranked, height=0.62, color=colors, zorder=3)
    axR.set_yticks(ypos)
    axR.set_yticklabels([f"候选 {i + 1}" for i in order], fontsize=8.2)
    axR.invert_yaxis()
    axR.set_xlim(0, max(ranked) * 1.55)
    axR.set_ylim(len(ranked) - 0.35, -1.55)
    axR.set_xticks([])
    axR.set_xlabel("代价（到目标表征的距离，越小越好）", fontsize=7.8, color=fs.MUTED, labelpad=3)
    axR.tick_params(axis="y", length=0, colors=fs.MUTED)
    for s in ("top", "right", "bottom"):
        axR.spines[s].set_visible(False)
    axR.spines["left"].set_color("#D1D5DB")
    axR.annotate("选中这条，只执行前几步", xy=(ranked[0] * 0.85, -0.02),
                 xytext=(ranked[0] * 1.10, -0.92),
                 fontsize=7.8, color=fs.ACCENT, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                 connectionstyle="arc3,rad=0.2"))
    axR.text(-0.055, 1.03, "(b)", transform=axR.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")

    return fs.save(fig, "basics", "ch08", "latent-planning")


if __name__ == "__main__":
    fig_action_jepa()
    fig_latent_planning()


# ---------------------------------------------------------------- 图 8-3 CEM 四轮
def _cem_toy(n_iter=4, n_samples=32, n_elite=4, sd0=0.9, seed=3):
    """二维玩具代价面上的 CEM 迭代，返回每一轮用到的采样分布与样本。"""
    th = np.deg2rad(32.0)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = rot @ np.diag([1.0, 6.0]) @ rot.T
    a_star = np.array([1.55, 0.85])

    def cost(p):
        q = p - a_star
        return np.einsum("...i,ij,...j->...", q, A, q)

    rng = np.random.default_rng(seed)
    mu = np.zeros(2)
    sd = np.array([sd0, sd0])
    rounds = []
    for _ in range(n_iter):
        S = rng.normal(mu, sd, size=(n_samples, 2))
        c = cost(S)
        order = np.argsort(c)
        elite = order[:n_elite]
        rounds.append(dict(mu=mu.copy(), sd=sd.copy(), S=S, cost=c, elite=elite))
        mu = S[elite].mean(axis=0)
        sd = S[elite].std(axis=0)
    return rounds, cost, a_star


def fig_cem():
    rounds, cost, a_star = _cem_toy()
    for i, r in enumerate(rounds):
        print("  round %d  mu (%.2f, %.2f)  sd (%.2f, %.2f)  best %.3f"
              % (i + 1, r["mu"][0], r["mu"][1], r["sd"][0], r["sd"][1], r["cost"].min()))

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.9))
    xs = np.linspace(-1.75, 3.25, 340)
    ys = np.linspace(-1.05, 2.45, 340)
    X, Y = np.meshgrid(xs, ys)
    Z = cost(np.stack([X, Y], axis=-1))
    levels = [0.1, 0.3, 0.7, 1.4, 2.5, 4.0, 6.0, 9.0, 13.0]

    for k, (ax, r) in enumerate(zip(axes.ravel(), rounds)):
        ax.contour(X, Y, Z, levels=levels, colors=["#D5DBE3"], linewidths=0.8, zorder=1)
        ax.contourf(X, Y, Z, levels=[0, 0.1], colors=["#FBE6D8"], zorder=0)

        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (r["mu"][0], r["mu"][1]), 2 * r["sd"][0], 2 * r["sd"][1],
            facecolor="none", edgecolor=fs.BLUE, lw=1.4, zorder=4))
        ax.scatter(r["mu"][0], r["mu"][1], s=44, marker="o", facecolor="white",
                   edgecolor=fs.BLUE, linewidth=1.3, zorder=6)

        mask = np.ones(len(r["S"]), dtype=bool)
        mask[r["elite"]] = False
        ax.scatter(r["S"][mask, 0], r["S"][mask, 1], s=12, color="#AAB2BE",
                   linewidth=0, zorder=3)
        ax.scatter(r["S"][r["elite"], 0], r["S"][r["elite"], 1], s=32,
                   color=fs.ACCENT, edgecolor="white", linewidth=0.6, zorder=5)

        ax.scatter(a_star[0], a_star[1], s=140, marker="*", color=fs.GOLD,
                   edgecolor="white", linewidth=0.9, zorder=7)

        ax.text(-1.62, 2.32, "第 %d 轮" % (k + 1), fontsize=9.4, color=fs.INK,
                weight="bold", ha="left", va="top", zorder=8)
        ax.text(-1.62, 2.04,
                "均值 (%.2f, %.2f)　标准差 (%.2f, %.2f)"
                % (r["mu"][0], r["mu"][1], r["sd"][0], r["sd"][1]),
                fontsize=8.2, color=fs.MUTED, ha="left", va="top", zorder=8)

        ax.set_xlim(-1.75, 3.25); ax.set_ylim(-1.05, 2.45)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1, 2, 3]); ax.set_yticks([-1, 0, 1, 2])
        ax.tick_params(labelsize=8.0, colors=fs.FAINT, length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#9CA3AF")
        ax.text(-0.02, 1.035, "(" + "abcd"[k] + ")", transform=ax.transAxes,
                fontsize=9.6, color=fs.MUTED, weight="bold", ha="left", va="bottom")
        if k >= 2:
            ax.set_xlabel("动作维度 1", fontsize=8.2, color=fs.MUTED, labelpad=2)
        if k % 2 == 0:
            ax.set_ylabel("动作维度 2", fontsize=8.2, color=fs.MUTED, labelpad=2)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.085,
                        wspace=0.10, hspace=0.22)
    fig.text(0.55, 0.012, "灰点为当轮 32 条候选，橙点为选中的 4 条，蓝色椭圆是一个标准差",
             fontsize=8.0, color=fs.MUTED, ha="center", va="bottom")
    return fs.save(fig, "basics", "ch08", "cem")


# ---------------------------------------------------------------- 图 8-4 MPC 时序
def fig_mpc_timeline():
    T = 4                      # 规划时域（步）
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.3),
                             gridspec_kw=dict(height_ratios=[1.25, 1.0], hspace=0.60))

    def panel(ax, k_exec, n_plan, title, note, inner, ybot):
        for j in range(n_plan):
            t0 = j * k_exec
            y = n_plan - j
            ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
                (t0 + 0.04, y - 0.27), T - 0.08, 0.54,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#FDF0E6", edgecolor=fs.ACCENT, lw=0.9, zorder=2))
            ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
                (t0 + 0.04, y - 0.27), k_exec - 0.08, 0.54,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=fs.ACCENT, edgecolor=fs.ACCENT, lw=0.9, alpha=0.92, zorder=3))
            ax.text(t0 + k_exec / 2, y, inner, fontsize=8.2, color="white",
                    ha="center", va="center", zorder=4)
            if T - k_exec >= 2.0:
                ax.text(t0 + k_exec + (T - k_exec) / 2, y, "只规划不执行", fontsize=8.2,
                        color=fs.ACCENT, ha="center", va="center", zorder=4)
            ax.scatter([t0], [y - 0.47], s=26, marker="o", color=fs.BLUE,
                       edgecolor="white", linewidth=0.7, zorder=5, clip_on=False)
            ax.plot([t0, t0], [y - 0.42, y - 0.27], color=fs.BLUE, lw=0.8, zorder=1)

        ax.set_xlim(-0.35, 11.2)
        ax.set_ylim(ybot, n_plan + 0.80)
        ax.set_yticks([n_plan - j for j in range(n_plan)])
        ax.set_yticklabels(["第 %d 次规划" % (j + 1) for j in range(n_plan)], fontsize=8.4)
        ax.set_xticks(np.arange(0, 11, 2))
        ax.tick_params(axis="x", labelsize=8.4)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#9CA3AF")
        ax.set_xlabel("环境步（蓝点为重新观测、重新编码的位置）", fontsize=8.2,
                      color=fs.MUTED, labelpad=3)
        ax.text(-0.008, 1.10, title, transform=ax.transAxes, fontsize=9.6,
                color=fs.INK, weight="bold", ha="left", va="bottom")
        ax.text(1.0, 1.10, note, transform=ax.transAxes, fontsize=8.2,
                color=fs.MUTED, ha="right", va="bottom")

    panel(axes[0], 1, 4, "(a) 每步重规划：k = 1", "走一步，重算一次", "执行", 0.35)
    panel(axes[1], 3, 2, "(b) 每三步重规划：k = 3", "规划次数减半，纠正也减半",
          "执行 3 步，中途不纠正", -0.78)

    ax = axes[0]
    ax.annotate("", xy=(0.12, 4.50), xytext=(3.88, 4.50),
                arrowprops=dict(arrowstyle="<->", color=fs.ACCENT, lw=0.9))
    ax.text(2.0, 4.56, "时域 $T$ = 4 步", fontsize=8.4, color=fs.ACCENT,
            ha="center", va="bottom")

    ax = axes[1]
    ax.annotate("蓝点之间的三步靠开环执行，漂移无人纠正", xy=(1.45, 1.72), xytext=(0.62, -0.30),
                fontsize=8.2, color=fs.PURPLE, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=fs.PURPLE, lw=0.9,
                                connectionstyle="arc3,rad=0.18", shrinkA=2, shrinkB=3))

    fig.subplots_adjust(left=0.145, right=0.985, top=0.925, bottom=0.115)
    return fs.save(fig, "basics", "ch08", "mpc-timeline")
