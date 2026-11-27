# -*- coding: utf-8 -*-
"""第 12 章插图：搜索树、beam search、CEM 采样、MCTS 四阶段。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import figstyle as fs

# ---------------------------------------------------------------------------
# 图 12-1　搜索树与两种展开顺序
# ---------------------------------------------------------------------------
def fig_search_tree():
    fig = plt.figure(figsize=(7.0, 3.5))
    axL = fig.add_axes([0.020, 0.050, 0.450, 0.905])
    axR = fig.add_axes([0.528, 0.070, 0.452, 0.860])
    for ax in (axL, axR):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    R = (0.50, 0.875)
    L1 = [(0.200, 0.610), (0.500, 0.610), (0.800, 0.610)]
    L2 = [(0.060, 0.320), (0.200, 0.320), (0.340, 0.320),
          (0.400, 0.320), (0.500, 0.320), (0.600, 0.320),
          (0.660, 0.320), (0.800, 0.320), (0.940, 0.320)]
    edges = [(R, p) for p in L1]
    for i, p in enumerate(L1):
        edges += [(p, c) for c in L2[3 * i:3 * i + 3]]

    GOAL = L2[4]
    path_edges = {(R, L1[1]), (L1[1], GOAL)}
    visited = set([R, L1[0], L1[1], L1[2], L2[0], L2[1], L2[2], L2[3]])
    frontier = {L2[4]}

    for a, b in edges:
        hot = (a, b) in path_edges
        if hot:
            col, lw = fs.ACCENT, 2.0
        elif a in visited and b in visited:
            col, lw = fs.INK, 1.1
        elif a in visited and b in frontier:
            col, lw = fs.ACCENT, 1.6
        else:
            col, lw = "#D6D9DE", 0.9
        axL.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw, zorder=2)

    for p in [R] + L1 + L2:
        if p == GOAL:
            axL.scatter([p[0]], [p[1]], s=170, facecolor="none", edgecolor=fs.ACCENT,
                        linewidth=0.9, zorder=5)
            axL.scatter([p[0]], [p[1]], s=58, facecolor="white", edgecolor=fs.ACCENT,
                        linewidth=1.4, zorder=6)
        elif p in visited:
            axL.scatter([p[0]], [p[1]], s=50, facecolor=fs.INK, edgecolor="white",
                        linewidth=0.8, zorder=6)
        else:
            axL.scatter([p[0]], [p[1]], s=50, facecolor="white", edgecolor="#9CA3AF",
                        linewidth=1.0, zorder=6)

    fs.label(axL, 0.500, 0.222, "目标", fs=7.2, color=fs.ACCENT)
    fs.arrow(axL, (0.500, 0.245), (0.500, 0.283), color=fs.ACCENT, lw=0.9, ms=7)

    # 路径上的代价
    fs.label(axL, 0.545, 0.745, "代价 1", fs=6.8, color=fs.ACCENT, ha="left")
    fs.label(axL, 0.545, 0.468, "代价 2", fs=6.8, color=fs.ACCENT, ha="left")

    for yy, txt in ((0.875, "第 0 层"), (0.610, "第 1 层"), (0.320, "第 2 层")):
        fs.label(axL, 0.985, yy + 0.072, txt, fs=6.8, color=fs.FAINT, ha="right")

    fs.label(axL, 0.010, 0.150, "实心是已展开的状态，空心是还没轮到的前沿", fs=7.2,
             color=fs.MUTED, ha="left")
    fs.label(axL, 0.010, 0.075, "每个状态 3 个后继，两层就是 1+3+9 = 13 个节点", fs=7.2,
             color=fs.MUTED, ha="left")
    axL.text(-0.020, 1.020, "(a)", transform=axL.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")

    # ---- 右：两种展开顺序
    depth_of = lambda p: 0 if p == R else (1 if p in L1 else 2)
    depth_col = ["#1F2937", "#1D4ED8", "#C2410C"]

    bfs = [R] + L1 + L2
    dfs = [R, L1[0], L2[0], L2[1], L2[2], L1[1], L2[3], L2[4], L2[5], L1[2],
           L2[6], L2[7], L2[8]]

    def chips(y, seq, title, note):
        fs.label(axR, 0.000, y + 0.165, title, fs=8.4, color=fs.INK, ha="left", weight="bold")
        fs.label(axR, 1.000, y + 0.165, note, fs=7.0, color=fs.MUTED, ha="right")
        w, gap = 0.0595, 0.0098
        for k, p in enumerate(seq):
            x = k * (w + gap)
            fs.box(axR, x, y - 0.052, w, 0.104, str(k + 1), fc="white",
                   ec=depth_col[depth_of(p)], tc=depth_col[depth_of(p)], fs=6.8, lw=1.0,
                   rounded=0.009)

    chips(0.660, bfs, "广度优先", "同时要记住 9 条分支")
    chips(0.270, dfs, "深度优先", "同时只记住 3 条分支")

    for k, (c, t) in enumerate(zip(depth_col, ["第 0 层", "第 1 层", "第 2 层"])):
        x = 0.005 + k * 0.290
        axR.add_patch(Polygon([[x, 0.030], [x + 0.026, 0.030], [x + 0.026, 0.058],
                               [x, 0.058]], closed=True, facecolor="white",
                              edgecolor=c, lw=1.0))
        fs.label(axR, x + 0.034, 0.044, t, fs=6.8, color=fs.MUTED, ha="left")
    fs.label(axR, 1.000, 0.044, "编号是访问次序", fs=6.8, color=fs.FAINT, ha="right")

    axR.text(-0.020, 1.020, "(b)", transform=axR.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")
    return fs.save(fig, "basics", "ch12", "search-tree")


# ---------------------------------------------------------------------------
# 图 12-2　beam search 保留三条前缀
# ---------------------------------------------------------------------------
def fig_beam():
    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_axes([0.025, 0.045, 0.605, 0.905])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    X = [0.050, 0.275, 0.500, 0.725, 0.950]
    kept = [
        [0.560],
        [0.800, 0.560, 0.320],
        [0.775, 0.535, 0.295],
        [0.750, 0.510, 0.270],
        [0.725, 0.485, 0.245],
    ]
    scores = [
        [0.0],
        [-0.42, -0.55, -0.71],
        [-0.84, -0.97, -1.15],
        [-1.22, -1.38, -1.57],
        [-1.60, -1.75, -1.92],
    ]

    for k in range(1, 5):
        fs.label(ax, X[k], 0.948, f"第 {k} 步", fs=6.8, color=fs.FAINT)

    # 被剪掉的分支：单独一条带
    ax.plot([0.000, 0.985], [0.155, 0.155], color="#E5E7EB", lw=0.9, zorder=1)
    fs.label(ax, 0.000, 0.088, "剪掉", fs=6.6, color="#9CA3AF", ha="left")

    ax.scatter([X[0]], [0.560], s=95, color=fs.INK, edgecolor="white", linewidth=1.0, zorder=8)
    fs.label(ax, 0.000, 0.625, "起始", fs=7.0, color=fs.MUTED, ha="left")

    for k in range(1, 5):
        for j in range(3):
            x, y = X[k], kept[k][j]
            py = kept[k - 1][min(j, len(kept[k - 1]) - 1)]
            ax.plot([X[k - 1] + 0.012, x - 0.012], [py, y], color=fs.INK, lw=0.95, zorder=2)
            ax.scatter([x], [y], s=95, color="white", edgecolor=fs.INK, linewidth=1.1, zorder=8)
            fs.label(ax, x, y - 0.088, f"{scores[k][j]:.2f}", fs=6.6, color=fs.MUTED)

        # 每步只画一条被剪掉的分支，落在下方隔离带
        ax.plot([X[k - 1] + 0.012, X[k] - 0.014], [kept[k - 1][-1], 0.118],
                color="#C9CDD4", lw=0.8, ls=(0, (2.5, 2.0)), zorder=1)
        ax.plot([X[k] - 0.026, X[k] + 0.002], [0.104, 0.132], color="#9CA3AF", lw=0.85, zorder=4)
        ax.plot([X[k] - 0.026, X[k] + 0.002], [0.132, 0.104], color="#9CA3AF", lw=0.85, zorder=4)
        fs.label(ax, X[k] + 0.012, 0.060, f"{scores[k][2] - 0.30:.2f}", fs=6.2,
                 color="#B0B5BD", ha="left")

    best = [(X[0], 0.560)] + [(X[k], kept[k][0]) for k in range(1, 5)]
    ax.plot([p[0] for p in best], [p[1] for p in best], color=fs.ACCENT, lw=2.2,
            zorder=6, solid_capstyle="round")
    for x, y in best[1:]:
        ax.scatter([x], [y], s=110, facecolor=fs.ACCENT, edgecolor="white",
                   linewidth=1.0, zorder=9)
    fs.label(ax, X[4] - 0.010, 0.800, "最终选这条", fs=7.2, color=fs.ACCENT, ha="right")
    fs.label(ax, 0.000, 0.020, "每一步只保留 3 条前缀，灰色叉号是被剪掉的候选", fs=6.8,
             color=fs.FAINT, ha="left")

    ax.text(-0.020, 1.020, "(a)", transform=ax.transAxes, fontsize=9.6, color=fs.MUTED,
            weight="bold", ha="left", va="bottom")

    # ---- 右：保留条数与最终质量
    axR = fig.add_axes([0.712, 0.205, 0.258, 0.665])
    labs = ["贪心\n1 条", "保留\n3 条", "保留\n8 条"]
    vals = [0.62, 0.78, 0.82]
    bars = axR.bar([0, 1, 2], vals, width=0.54,
                   color=["#CBD5E1", fs.ACCENT, "#E9C8B6"], zorder=3)
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.030, f"{v:.2f}", ha="center",
                 fontsize=7.2, color=fs.INK)
    axR.set_xticks([0, 1, 2]); axR.set_xticklabels(labs, fontsize=7.0, color=fs.MUTED)
    axR.set_xlim(-0.65, 2.65)
    axR.set_ylim(0, 1.02); axR.set_yticks([])
    axR.minorticks_off()
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)
    axR.spines["bottom"].set_color("#D1D5DB")
    axR.tick_params(axis="both", length=0)
    axR.set_ylabel("最终得分（示意）", fontsize=7.2, color=fs.MUTED, labelpad=5)
    axR.text(-0.30, 1.075, "(b)", transform=axR.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")
    return fs.save(fig, "basics", "ch12", "beam")


# ---------------------------------------------------------------------------
# 图 12-3　CEM 的采样与收缩
# ---------------------------------------------------------------------------
def _cem_rounds(goal, A, mu0, cov0, n=200, k=20, rounds=4, seed=7):
    rng = np.random.default_rng(seed)
    mu, cov = np.asarray(mu0, float), np.asarray(cov0, float)
    hist = []
    for r in range(rounds):
        pts = rng.multivariate_normal(mu, cov, size=n)
        cost = np.einsum("ni,ij,nj->n", pts - goal, A, pts - goal)
        idx = np.argsort(cost)[:k]
        hist.append((mu.copy(), cov.copy(), pts, pts[idx]))
        mu = pts[idx].mean(axis=0)
        cov = np.cov(pts[idx].T) * 1.35
    return hist, mu, cov


def fig_cem():
    goal = np.array([6.55, 6.30])
    A = np.array([[0.115, -0.055], [-0.055, 0.150]])
    hist, mu_end, cov_end = _cem_rounds(goal, A, (2.30, 8.40), [[4.10, 0.95], [0.95, 2.30]])

    for i, (mu, cov, pts, el) in enumerate(hist):
        print(f"  CEM 第 {i + 1} 轮 均值 {mu.round(2)} 方差迹 {np.trace(cov):.2f}")
    print(f"  CEM 第 5 轮 均值 {mu_end.round(2)} 方差迹 {np.trace(cov_end):.2f}")

    fig = plt.figure(figsize=(7.0, 3.5))
    axL = fig.add_axes([0.060, 0.135, 0.420, 0.790])
    axR = fig.add_axes([0.600, 0.185, 0.365, 0.715])

    xs = np.linspace(-0.6, 12.6, 300)
    ys = np.linspace(-0.6, 12.6, 300)
    XX, YY = np.meshgrid(xs, ys)
    DX, DY = XX - goal[0], YY - goal[1]
    C = A[0, 0] * DX ** 2 + 2 * A[0, 1] * DX * DY + A[1, 1] * DY ** 2
    axL.contour(XX, YY, np.ma.masked_greater(C, 13.0), levels=[0.5, 1.6, 3.6, 7.0, 12.0],
                colors=["#DEE0E4"] * 5, linewidths=0.7, zorder=1)

    # 第一轮的全部样本与精英
    mu, cov, pts, el = hist[0]
    axL.scatter(pts[:, 0], pts[:, 1], s=6, color="#9CA3AF", alpha=0.70, zorder=3)
    axL.scatter(el[:, 0], el[:, 1], s=30, facecolor="white", edgecolor=fs.ACCENT,
                linewidth=0.9, zorder=6)

    cols = ["#C3CDD8", "#8FA3B6", "#5C7488", fs.ACCENT]
    ell_handles = []
    for i in range(4):
        m = hist[i][0] if i < len(hist) else mu_end
        c = hist[i][1] if i < len(hist) else cov_end
        w, h = 2 * np.sqrt(np.linalg.eigvalsh(c))[::-1] * 1.5
        ang = np.degrees(0.5 * np.arctan2(2 * c[0, 1], c[0, 0] - c[1, 1]))
        ls = "-" if i == 0 else (0, (4.5, 2.2))
        axL.add_patch(Ellipse(m, w, h, angle=ang, facecolor="none", edgecolor=cols[i],
                              lw=1.7 if i == 0 else 1.3, ls=ls, zorder=5))
        ell_handles.append(Line2D([], [], color=cols[i], lw=1.5, ls=ls))

    axL.scatter([goal[0]], [goal[1]], s=150, marker="*", color=fs.BLUE,
                edgecolor="white", linewidth=0.9, zorder=9)
    fs.label(axL, goal[0] + 0.70, goal[1] - 1.15, "代价最小处", fs=7.4, color=fs.BLUE, ha="left")

    axL.set_xlim(-0.5, 12.4); axL.set_ylim(-0.5, 12.6)
    axL.set_xticks([]); axL.set_yticks([])
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axL.spines[s].set_color("#D1D5DB")
    axL.set_xlabel("动作维度 1", fontsize=7.8, color=fs.MUTED, labelpad=2)
    axL.set_ylabel("动作维度 2", fontsize=7.8, color=fs.MUTED, labelpad=2)

    handles = ell_handles + [Line2D([], [], marker="o", ls="none", markerfacecolor="white",
                                   markeredgecolor=fs.ACCENT, markeredgewidth=0.9,
                                   markersize=5)]
    axL.legend(handles, ["第 1 轮", "第 2 轮", "第 3 轮", "第 4 轮", "精英"],
               loc="lower left", bbox_to_anchor=(0.005, 0.005), fontsize=6.4,
               ncol=2, handlelength=2.2, columnspacing=1.0, labelspacing=0.50,
               frameon=True, framealpha=1.0, edgecolor="#E5E7EB", borderpad=0.60)
    axL.text(-0.115, 1.030, "(a)", transform=axL.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")

    # ---- 右：维数与采样量
    dims = [2, 3, 5, 7]
    need = [3 ** d for d in dims]
    axR.bar([0, 1, 2, 3], need, width=0.56,
            color=["#CBD5E1", "#B3C4D2", "#96A9BA", fs.ACCENT], zorder=3)
    axR.set_yscale("log")
    for i, v in enumerate(need):
        axR.text(i, v * 1.55, f"{v}", ha="center", fontsize=7.4, color=fs.INK)
    axR.set_xticks([0, 1, 2, 3]); axR.set_xticklabels([f"{d} 维" for d in dims],
                                                     fontsize=7.4, color=fs.MUTED)
    axR.set_ylim(1, 1.6e4)
    axR.set_yticks([]); axR.minorticks_off()
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)
    axR.spines["bottom"].set_color("#D1D5DB")
    axR.tick_params(axis="both", length=0, which="both")
    axR.set_ylabel("需要的样本数（每维 3 个格点，示意）", fontsize=7.0, color=fs.MUTED, labelpad=5)
    axR.text(-0.30, 1.075, "(b)", transform=axR.transAxes, fontsize=9.6, color=fs.MUTED,
             weight="bold", ha="left", va="bottom")
    return fs.save(fig, "basics", "ch12", "cem-sampling")


# ---------------------------------------------------------------------------
# 图 12-4　MCTS 的四个阶段
# ---------------------------------------------------------------------------
def fig_mcts():
    fig = plt.figure(figsize=(7.2, 3.6))
    axes = [fig.add_axes([0.012, 0.070, 0.238, 0.860]),
            fig.add_axes([0.256, 0.070, 0.238, 0.860]),
            fig.add_axes([0.500, 0.070, 0.238, 0.860]),
            fig.add_axes([0.744, 0.070, 0.246, 0.860])]

    P = {
        "R":  (0.50, 0.880),
        "A":  (0.24, 0.630), "B": (0.76, 0.630),
        "A1": (0.07, 0.400), "A2": (0.41, 0.400),
        "B1": (0.59, 0.400), "B2": (0.93, 0.400),
        "X":  (0.93, 0.200),
    }
    E = [("R", "A"), ("R", "B"), ("A", "A1"), ("A", "A2"),
         ("B", "B1"), ("B", "B2")]
    VAL = {"R": (20, 0.52), "A": (8, 0.44), "B": (12, 0.61),
           "A1": (3, 0.30), "A2": (5, 0.52), "B1": (6, 0.55), "B2": (6, 0.68)}

    def new_ax(i):
        ax = axes[i]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        return ax

    def edge(ax, a, b, col=fs.INK, lw=1.0, ls="-", z=2):
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color=col, lw=lw, ls=ls, zorder=z)

    BOX = dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96)

    def node(ax, k, fill="white", ec=fs.INK, tc=fs.INK, s=155, lw=1.1, val=None,
             side="below", labcol=None):
        x, y = P[k]
        ax.scatter([x], [y], s=s, facecolor=fill, edgecolor=ec, linewidth=lw, zorder=7)
        n, v = val if val else VAL[k]
        fs.label(ax, x, y + 0.008, f"{n}", fs=6.8, color=tc, z=10)
        if side == "below":
            fs.label(ax, x, y - 0.062, f"{v:.2f}", fs=6.2, color=labcol or fs.MUTED, z=9,
                     bbox=BOX)
        elif side == "left":
            fs.label(ax, x - 0.048, y + 0.002, f"{v:.2f}", fs=6.2, color=labcol or fs.MUTED,
                     ha="right", z=9, bbox=BOX)
        else:
            fs.label(ax, x + 0.048, y + 0.002, f"{v:.2f}", fs=6.2, color=labcol or fs.MUTED,
                     ha="left", z=9, bbox=BOX)

    def base(ax, extra=()):
        for a, b in E:
            edge(ax, a, b, "#CED2D8", 0.9)
        for k in ["A", "A1", "A2", "B1"]:
            node(ax, k, fill="#F3F4F6", ec="#A4ABB5", tc=fs.MUTED, s=130, lw=0.9)
        for a, b in [("R", "B"), ("B", "B2")]:
            edge(ax, a, b, fs.ACCENT, 2.0, z=4)

    # --- (a) 选择
    ax = new_ax(0)
    base(ax)
    node(ax, "R", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5)
    node(ax, "B", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5, side="right")
    node(ax, "B2", fill=fs.ACCENT, ec=fs.ACCENT, tc="white", lw=1.1, side="left")
    ucbA = 0.44 + np.sqrt(np.log(20) / 8)
    ucbB = 0.61 + np.sqrt(np.log(20) / 12)
    fs.label(ax, 0.000, 0.248, f"B 的上界 0.61 + {np.sqrt(np.log(20) / 12):.2f} = {ucbB:.2f}",
             fs=6.4, color=fs.ACCENT, ha="left")
    fs.label(ax, 0.000, 0.180, f"A 的上界 0.44 + {np.sqrt(np.log(20) / 8):.2f} = {ucbA:.2f}",
             fs=6.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.500, 0.045, "每次挑上界最高的孩子", fs=6.8, color=fs.MUTED)

    # --- (b) 扩展
    ax = new_ax(1)
    base(ax)
    node(ax, "R", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5)
    node(ax, "B", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5, side="right")
    node(ax, "B2", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5, side="left")
    edge(ax, "B2", "X", fs.GREEN, 1.5, ls=(0, (3.2, 2.2)), z=5)
    ax.scatter([P["X"][0]], [P["X"][1]], s=140, facecolor=fs.GREEN, edgecolor=fs.GREEN,
               linewidth=1.0, zorder=7)
    fs.label(ax, P["X"][0], P["X"][1] + 0.014, "0", fs=6.8, color="white", z=9)
    fs.label(ax, P["X"][0] - 0.010, P["X"][1] - 0.055, "新节点", fs=6.2, color=fs.GREEN,
             ha="right")
    fs.label(ax, 0.500, 0.045, "第一次走到这里，添一个孩子", fs=6.8, color=fs.MUTED)

    # --- (c) 模拟
    ax = new_ax(2)
    base(ax)
    node(ax, "R", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5)
    node(ax, "B", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5, side="right")
    node(ax, "B2", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5, side="left")
    ax.scatter([P["X"][0]], [P["X"][1]], s=140, facecolor=fs.GREEN, edgecolor=fs.GREEN,
               linewidth=1.0, zorder=7)
    fs.label(ax, P["X"][0], P["X"][1] + 0.014, "0", fs=6.8, color="white", z=9)

    sim = [(0.800, 0.128), (0.700, 0.062)]
    prev = P["X"]
    for i, q in enumerate(sim):
        ax.plot([prev[0], q[0]], [prev[1], q[1]], color=fs.GREEN, lw=1.0, ls=(0, (2.5, 2)),
                zorder=4)
        if i < len(sim) - 1:
            ax.scatter([q[0]], [q[1]], s=24, facecolor="white", edgecolor=fs.GREEN, lw=0.9,
                       zorder=6)
        prev = q
    ax.scatter([0.700], [0.062], s=52, marker="s", facecolor="white", edgecolor=fs.GREEN,
               lw=1.1, zorder=6)
    fs.label(ax, 0.665, 0.038, "终局", fs=6.2, color=fs.GREEN, ha="right")
    fs.label(ax, 0.010, 0.130, "乱走几步", fs=6.6, color=fs.GREEN, ha="left")
    fs.label(ax, 0.010, 0.088, "得到回报 0.9", fs=6.6, color=fs.GREEN, ha="left")

    # --- (d) 回传
    ax = new_ax(3)
    base(ax)
    edge(ax, "B2", "X", fs.ACCENT, 1.6, z=4)
    newv = {}
    for k in ["X", "B2", "B", "R"]:
        n0, v0 = (0, 0.0) if k == "X" else VAL[k]
        newv[k] = (n0 + 1, (n0 * v0 + 0.9) / (n0 + 1))
    node(ax, "R", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5, val=newv["R"])
    node(ax, "B", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5, side="right", val=newv["B"])
    node(ax, "B2", ec=fs.ACCENT, tc=fs.ACCENT, lw=1.5, side="left", val=newv["B2"])
    ax.scatter([P["X"][0]], [P["X"][1]], s=140, facecolor=fs.ACCENT, edgecolor=fs.ACCENT,
               linewidth=1.0, zorder=7)
    fs.label(ax, P["X"][0], P["X"][1] + 0.014, "1", fs=6.8, color="white", z=9)
    fs.label(ax, P["X"][0], P["X"][1] - 0.058, "0.90", fs=6.4, color=fs.MUTED, z=9)

    for a, b in [("X", "B2"), ("B2", "B"), ("B", "R")]:
        x0, y0 = P[a][0], P[a][1]
        x1, y1 = P[b][0], P[b][1]
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy)
        ox, oy = -dy / L * 0.045, dx / L * 0.045
        fs.arrow(ax, (x0 + dx * 0.30 + ox, y0 + dy * 0.30 + oy),
                 (x0 + dx * 0.70 + ox, y0 + dy * 0.70 + oy), color=fs.PURPLE, lw=1.1, ms=8)
    fs.label(ax, 0.010, 0.130, "沿路的节点", fs=6.6, color=fs.PURPLE, ha="left")
    fs.label(ax, 0.010, 0.088, "次数加一", fs=6.6, color=fs.PURPLE, ha="left")
    fs.label(ax, 0.500, 0.020, "均值按新的回报重算", fs=6.6, color=fs.MUTED)

    for ax, tag in zip(axes, ["选择", "扩展", "模拟", "回传"]):
        ax.text(-0.015, 1.015, tag, transform=ax.transAxes, fontsize=9.0, color=fs.INK,
                weight="bold", ha="left", va="bottom")

    return fs.save(fig, "basics", "ch12", "mcts")


def report():
    """正文里用到的算术，全部由这里算出，避免手写数字出错。"""
    print("b=3,d=2 节点数 ->", sum(3 ** k for k in range(3)))
    print("b=3,d=8 节点数 ->", sum(3 ** k for k in range(9)))
    print("b=10,d=10 节点数 ->", 10 ** 10)
    print("3^d d=2,3,5,7 ->", [3 ** d for d in (2, 3, 5, 7)])
    print("ucb A,B ->", round(0.44 + np.sqrt(np.log(20) / 8), 3),
          round(0.61 + np.sqrt(np.log(20) / 12), 3))


if __name__ == "__main__":
    fig_search_tree()
    fig_beam()
    fig_cem()
    fig_mcts()
    report()
