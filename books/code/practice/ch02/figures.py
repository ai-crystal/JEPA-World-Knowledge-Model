# -*- coding: utf-8 -*-
"""第 2 章插图：三种塌缩状态、奇异值谱对比、塌缩时的三条监控曲线。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs


def fig_states():
    """同一坐标范围下的三种状态：正常、完全塌缩、维度塌缩。数据为脚本生成的示意数据。"""
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.95))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.175, top=0.905, wspace=0.28)

    centers = np.array([[-1.15, 0.90], [1.05, 1.10], [0.10, -1.30]])
    cols = [fs.BLUE, fs.GREEN, fs.ACCENT]
    n = 55
    u = np.array([1.0, 0.52])
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])

    # (a) 正常：三个类各自成团，两个方向都有伸展
    shear = np.array([[1.0, 0.22], [0.0, 0.92]])
    pts_normal = [centers[k] + rng.normal(0, 0.30, (n, 2)) @ shear.T for k in range(3)]

    # (b) 完全塌缩：所有点落在原点
    pts_full = [np.zeros((n, 2)) for _ in range(3)]

    # (c) 维度塌缩：三个类沿同一条直线排开，其余方向被压平
    pts_dim = []
    for k, t0 in enumerate((-1.55, 0.0, 1.55)):
        t = t0 + rng.normal(0, 0.38, n)
        pts_dim.append(np.outer(t, u) + np.outer(rng.normal(0, 0.055, n), v))

    titles = ["(a) 正常", "(b) 完全塌缩", "(c) 维度塌缩"]
    subs = ["两个方向都在变化", "所有点缩成一处", "只剩一个方向在变化"]

    for i, ax in enumerate(axes):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(length=2.6, labelsize=7.8, pad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(fs.FAINT)
        ax.set_title(titles[i], fontsize=9.6, color=fs.INK, pad=5)
        ax.set_xlabel(subs[i], fontsize=8.3, color=fs.MUTED, labelpad=4)

    # (a)
    for k in range(3):
        axes[0].scatter(pts_normal[k][:, 0], pts_normal[k][:, 1], s=11, color=cols[k],
                        alpha=0.85, linewidths=0, zorder=3)

    # (b)
    axes[1].scatter([0], [0], s=58, color=fs.INK, linewidths=0, zorder=4)
    axes[1].annotate("所有点重合", xy=(0.06, 0.06), xytext=(0.75, 1.55),
                     fontsize=8.2, color=fs.MUTED, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=fs.FAINT, lw=0.9,
                                     shrinkA=1, shrinkB=3))

    # (c)
    axes[2].plot([-3.15 * u[0], 3.15 * u[0]], [-3.15 * u[1], 3.15 * u[1]],
                 color=fs.FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for k in range(3):
        axes[2].scatter(pts_dim[k][:, 0], pts_dim[k][:, 1], s=11, color=cols[k],
                        alpha=0.85, linewidths=0, zorder=3)

    return fs.save(fig, "practice", "ch02", "three-states")


def fig_spectrum():
    """三条奇异值谱：健康、维度塌缩、完全塌缩。曲线按典型形状合成，不是论文读数。"""
    d = 2048
    k = np.arange(1, d + 1)
    t = k / d

    healthy = 8.6 - 2.6 * t ** 0.55
    collapse = np.maximum(8.6 - 0.75 * np.sqrt(k), -2.0)
    full = np.full_like(t, -6.0)

    def eff_rank(sigma):
        p = sigma / sigma.sum()
        return float(np.exp(-(p * np.log(p + 1e-300)).sum()))

    er_healthy = eff_rank(10 ** healthy)
    er_collapse = eff_rank(10 ** collapse)
    er_full = eff_rank(10 ** full)
    print(f"  演示谱线有效秩: 健康 {er_healthy:.1f}, 维度塌缩 {er_collapse:.1f}, 完全塌缩 {er_full:.1f}")

    fig = plt.figure(figsize=(6.9, 3.55))
    ax = fig.add_axes([0.088, 0.145, 0.888, 0.815])
    ax.set_xlim(0, d)
    ax.set_ylim(-7.4, 9.4)

    ax.plot(k, healthy, color=fs.BLUE, lw=1.9, zorder=4)
    ax.plot(k, collapse, color=fs.ACCENT, lw=1.9, zorder=5)
    ax.plot(k, full, color=fs.FAINT, lw=1.9, ls=(0, (5, 2.5)), zorder=3)

    ax.axhline(0.0, color=fs.FAINT, lw=0.8, ls=(0, (1.5, 2.5)), zorder=1)
    ax.text(2050, -0.75, "对数零点（奇异值为 1）", fontsize=7.6, color=fs.FAINT, ha="right", va="top")

    fs.clean(ax, grid=True)
    ax.set_xlabel("奇异值序号（按从大到小排列）", fontsize=9.4, color=fs.INK, labelpad=5)
    ax.set_ylabel("奇异值的对数", fontsize=9.4, color=fs.INK, labelpad=6)
    ax.set_xticks([0, 512, 1024, 1536, 2048])
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6, 8])
    ax.tick_params(labelsize=8.4, length=3, colors=fs.MUTED)

    handles = [Line2D([], [], color=fs.BLUE, lw=1.9, label="有投影头"),
               Line2D([], [], color=fs.ACCENT, lw=1.9, label="无投影头"),
               Line2D([], [], color=fs.FAINT, lw=1.9, ls=(0, (5, 2.5)), label="等幅")]
    leg = ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.665, 0.545),
                    handlelength=2.0, handletextpad=0.7, labelspacing=0.55,
                    borderpad=0.2, fontsize=9.0)
    for txt in leg.get_texts():
        txt.set_color(fs.INK)

    fs.label(ax, 300, -3.75, "两百维之后压到 1 以下，其余维度几乎不再变化", fs=8.6,
             color=fs.ACCENT, ha="left")
    ax.annotate("", xy=(214, -1.92), xytext=(240, -3.35),
                arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                shrinkA=3, shrinkB=2))

    fs.label(ax, 2015, -7.0, "演示数据，非论文实测读数", fs=7.8, color=fs.FAINT, ha="right")

    return fs.save(fig, "practice", "ch02", "spectrum")


def fig_monitors():
    """停梯度有无时的三条曲线：训练损失、通道标准差、kNN 准确率。按论文曲线的形状重绘。"""
    d = 2048
    target = 1.0 / np.sqrt(d)
    ep = np.linspace(0, 100, 240)

    loss_ok = -1.0 + 0.95 * np.exp(-ep / 26.0) + 0.06 * np.exp(-ep / 3.0)
    loss_bad = -1.0 + 0.60 * np.exp(-ep / 1.1) + 0.02

    std_ok = target * (0.72 + 0.30 * (1 - np.exp(-ep / 8.0)))
    std_bad = target * 0.42 * np.exp(-ep / 2.2) + 0.0004

    knn_ok = 56.0 * (1 - np.exp(-ep / 34.0))
    knn_bad = np.full_like(ep, 0.35) + 0.25 * np.exp(-ep / 40.0)

    print(f"  1/sqrt(d) = {target:.4f}（d={d}）")

    fig = plt.figure(figsize=(7.0, 3.25))
    axes = [fig.add_axes([0.075, 0.255, 0.252, 0.705]),
            fig.add_axes([0.400, 0.255, 0.252, 0.705]),
            fig.add_axes([0.725, 0.255, 0.252, 0.705])]

    # (a) 训练损失
    ax = axes[0]
    ax.plot(ep, loss_ok, color=fs.BLUE, lw=1.8, zorder=4)
    ax.plot(ep, loss_bad, color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), zorder=4)
    ax.set_ylim(-1.12, 0.10)
    ax.set_yticks([-1.0, -0.5, 0.0])
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("训练损失", fontsize=9.0, labelpad=5)
    fs.label(ax, 4, -0.44, "两条曲线都收敛了", fs=8.4, color=fs.MUTED, ha="left")

    # (b) 通道标准差
    ax = axes[1]
    ax.plot(ep, std_ok, color=fs.BLUE, lw=1.8, zorder=4)
    ax.plot(ep, std_bad, color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), zorder=4)
    ax.axhline(target, color=fs.FAINT, lw=0.9, ls=(0, (1.5, 2.5)), zorder=1)
    ax.set_ylim(0, 0.030)
    ax.set_yticks([0, 0.01, 0.02, 0.03])
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("各通道标准差", fontsize=9.0, labelpad=5)
    fs.label(ax, 4, 0.0268, r"$1/\sqrt{d}$", fs=9.2, color=fs.MUTED, ha="left")
    fs.label(ax, 20, 0.0072, "一路滑到零", fs=8.4, color=fs.ACCENT, ha="left")

    # (c) kNN
    ax = axes[2]
    ax.plot(ep, knn_ok, color=fs.BLUE, lw=1.8, zorder=4)
    ax.plot(ep, knn_bad, color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), zorder=4)
    ax.set_ylim(0, 66)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("kNN 准确率（%）", fontsize=9.0, labelpad=5)
    fs.label(ax, 4, 12.5, "贴着零线", fs=8.4, color=fs.ACCENT, ha="left")

    tags = ["(a)", "(b)", "(c)"]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(labelsize=8.2, length=2.8, colors=fs.MUTED)
        fs.clean(ax, grid=True)
        ax.set_title(tags[i], fontsize=9.4, color=fs.INK, pad=4, loc="left", weight="bold")

    handles = [Line2D([], [], color=fs.BLUE, lw=1.8, label="有停梯度"),
               Line2D([], [], color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), label="去掉停梯度")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False, fontsize=9.0, handlelength=2.2,
               handletextpad=0.7, columnspacing=2.4)

    return fs.save(fig, "practice", "ch02", "monitors")


# ---------- 图 2-4 / 图 2-5 共用的谱线模型 ----------
# 谱线按幂律形状 log10(sigma_k) = A - B * (k/d)**0.62 生成，A 控制整体高度，B 控制尾部被压的
# 快慢。三条路径从同一个初始化（A=1.6, B=4.0）出发，只改终点，恰好对应正文里“结构相同、
# 目标不同”的对照。数值全部由本脚本算出，正文引用的读数来自下面的打印。

D_EMB = 2048
ENDS = {"healthy": (6.2, 2.65), "dim": (6.2, 13.5), "full": (-5.5, 0.04)}
START = (1.6, 4.0)


def _spec(A, B, d=D_EMB, q=0.62):
    x = np.arange(1, d + 1) / d
    return 10.0 ** (A - B * x ** q)


def _eff(s):
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-300)).sum()))


def _path(regime, t):
    """t 取 0 到 1，返回该训练进度下的谱线。"""
    A1, B1 = ENDS[regime]
    s = (1 - np.exp(-3.0 * t)) / (1 - np.exp(-3.0))
    A = START[0] + (A1 - START[0]) * s
    B = START[1] + (B1 - START[1]) * s
    return _spec(A, B)


def _tail_share(s):
    """后一半维度（第 d/2+1 到 d 个）占总能量的比例。"""
    return float(s[D_EMB // 2:].sum() / s.sum())


def report_numbers():
    """正文引用的读数：中心化的代价、噪声底、纯噪声的有效秩、秩与有效秩的差距。

    全部在这里现算，正文里的数字都来自本函数的打印，换参数就一起变。
    """
    rng = np.random.default_rng(7)

    # 1) 减不减均值：2048 维、单维标准差 0.02、均值偏到 0.05，4096 个样本
    d, n = 2048, 4096
    z = rng.normal(0, 0.02, size=(n, d)) + 0.05
    raw = z.T @ z / n
    cen = (z - z.mean(axis=0)).T @ (z - z.mean(axis=0)) / n
    print(f"  中心化：未减均值 迹 {np.trace(raw):.3f} 有效秩 "
          f"{_eff(np.linalg.svd(raw, compute_uv=False)):.2f}；"
          f"减均值后 迹 {np.trace(cen):.3f} 有效秩 "
          f"{_eff(np.linalg.svd(cen, compute_uv=False)):.1f}")

    # 2) 纯噪声的有效秩随样本数与维度的比值变化（d=512，重复五次取平均）
    d2 = 512
    for g in (0.125, 0.25, 0.5, 1.0, 2.0):
        m = int(round(d2 / g))
        vals = []
        for _ in range(5):
            q = rng.normal(size=(m, d2))
            q = q - q.mean(axis=0)
            vals.append(_eff(np.linalg.svd(q.T @ q / m, compute_uv=False)))
        print(f"  纯噪声 gamma={g:.3f}（样本 {m}）：有效秩 {np.mean(vals):.1f}，"
              f"占维度 {np.mean(vals) / d2:.2f}")

    # 3) 秩与有效秩：真实结构 32 维，混入 1e-3 量级的噪声
    z = rng.normal(size=(4096, 32)) @ rng.normal(size=(32, d)) * 0.3
    z = z + rng.normal(0, 1e-3, size=(4096, d))
    z = z - z.mean(axis=0)
    s = np.linalg.svd(z.T @ z / len(z), compute_uv=False)
    print(f"  32 维结构：矩阵秩 {np.linalg.matrix_rank(z.T @ z / len(z))}，"
          f"有效秩 {_eff(s):.1f}，第 33 个奇异值 {s[32]:.2e}，"
          f"相对首项 {s[32] / s[0]:.2e}")

    # 4) 噪声底的跨度上界 (1+sqrt(gamma))^2/(1-sqrt(gamma))^2
    for g in (0.25, 0.5, 1.0):
        lo = (1 - np.sqrt(g)) ** 2
        hi = (1 + np.sqrt(g)) ** 2
        span = "inf" if lo <= 1e-12 else f"{hi / lo:.1f}"
        print(f"  噪声底 gamma={g}: 下沿 {lo:.3f} 上沿 {hi:.3f} 跨度 {span}")


def fig_rank_curve():
    """有效秩随训练的变化，以及训练终点三条谱线的高度对比。曲线由谱线模型算出。"""
    ep = np.linspace(0, 100, 201)
    cols = {"healthy": fs.BLUE, "dim": fs.ACCENT, "full": fs.GREEN}
    names = {"healthy": "健康", "dim": "维度塌缩", "full": "完全塌缩"}

    curves = {r: np.array([_eff(_path(r, t)) for t in ep / 100.0]) for r in ENDS}
    for r in ENDS:
        s1 = _path(r, 1.0)
        print(f"  {names[r]}: 有效秩 {curves[r][-1]:.1f}, 尾部能量占比 {_tail_share(s1) * 100:.3g}%")

    fig = plt.figure(figsize=(7.0, 3.6))
    ax1 = fig.add_axes([0.075, 0.165, 0.395, 0.755])
    ax2 = fig.add_axes([0.585, 0.165, 0.395, 0.755])

    # (a) 有效秩
    for r in ENDS:
        lw = 1.9
        ls = "-" if r == "healthy" else (0, (5, 2.5)) if r == "dim" else (0, (1.5, 2.2))
        ax1.plot(ep, curves[r], color=cols[r], lw=lw, ls=ls, zorder=4)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 2200)
    ax1.set_yticks([0, 512, 1024, 1536, 2048])
    ax1.set_xticks([0, 50, 100])
    fs.clean(ax1, grid=True)
    ax1.set_xlabel("轮次", fontsize=9.2, labelpad=4)
    ax1.set_ylabel("有效秩（满值 2048）", fontsize=9.2, labelpad=5)
    ax1.tick_params(labelsize=8.2, length=2.8, colors=fs.MUTED)
    ax1.set_title("(a)", fontsize=9.4, color=fs.INK, pad=4, loc="left", weight="bold")
    fs.label(ax1, 62, 1240, "最高的一条反而是坏的", fs=8.6, color=fs.MUTED, ha="center")
    fs.label(ax1, 97, 2140, "完全塌缩", fs=8.6, color=fs.GREEN, ha="right")
    fs.label(ax1, 97, 790, "健康", fs=8.6, color=fs.BLUE, ha="right")
    fs.label(ax1, 97, 120, "维度塌缩", fs=8.6, color=fs.ACCENT, ha="right", va="bottom")

    # (b) 终点谱线
    k = np.arange(1, D_EMB + 1)
    for r in ENDS:
        ls = "-" if r == "healthy" else (0, (5, 2.5)) if r == "dim" else (0, (1.5, 2.2))
        ax2.plot(k, np.log10(_path(r, 1.0)), color=cols[r], lw=1.8, ls=ls, zorder=4)
    ax2.set_xlim(0, D_EMB)
    ax2.set_ylim(-9.0, 7.6)
    ax2.set_xticks([0, 1024, 2048])
    ax2.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6])
    fs.clean(ax2, grid=True)
    ax2.set_xlabel("奇异值序号", fontsize=9.2, labelpad=4)
    ax2.set_ylabel("奇异值的对数", fontsize=9.2, labelpad=5)
    ax2.tick_params(labelsize=8.2, length=2.8, colors=fs.MUTED)
    ax2.set_title("(b)", fontsize=9.4, color=fs.INK, pad=4, loc="left", weight="bold")
    ax2.annotate("", xy=(1150, -4.2), xytext=(1150, 3.5),
                 arrowprops=dict(arrowstyle="<->", color=fs.FAINT, lw=0.9))
    fs.label(ax2, 1240, -0.4, "同序号相差\n六到七个数量级", fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax2, 1980, 4.3, "健康", fs=8.6, color=fs.BLUE, ha="right")
    fs.label(ax2, 1980, -8.4, "维度塌缩", fs=8.6, color=fs.ACCENT, ha="right")
    fs.label(ax2, 60, -6.35, "完全塌缩（整条压平）", fs=8.6, color=fs.GREEN, ha="left")

    return fs.save(fig, "practice", "ch02", "rank-curve")


def fig_monitor_panel():
    """四联监控面板：逐层标准差、有效秩、尾部能量占比、线性探针。曲线为模型算出。"""
    d = D_EMB
    ref = 1.0 / np.sqrt(d)
    ep = np.linspace(0, 100, 201)

    fig = plt.figure(figsize=(7.2, 4.85))
    axes = [fig.add_axes([0.098, 0.575, 0.360, 0.330]),
            fig.add_axes([0.610, 0.575, 0.360, 0.330]),
            fig.add_axes([0.098, 0.110, 0.360, 0.330]),
            fig.add_axes([0.610, 0.110, 0.360, 0.330])]

    # (a) 逐层标准差
    ax = axes[0]
    layers = np.arange(1, 13)
    ok = ref * (0.86 + 0.16 * np.linspace(0, 1, 12))
    bad = ref * 0.80 * 0.62 ** (layers - 1)
    ax.semilogy(layers, ok, color=fs.BLUE, lw=1.7, marker="o", ms=3.2, zorder=4)
    ax.semilogy(layers, bad, color=fs.ACCENT, lw=1.7, marker="s", ms=2.9,
                ls=(0, (5, 2.5)), zorder=4)
    ax.axhspan(0.5 * ref, 2 * ref, color=fs.GRID, alpha=0.75, zorder=1, lw=0)
    ax.axhline(0.5 * ref, color=fs.MUTED, lw=0.9, ls=(0, (1.5, 2.2)), zorder=2)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(4e-5, 9e-2)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_yticks([1e-4, 1e-3, 1e-2])
    fs.clean(ax, grid=True)
    ax.set_xlabel("层号（浅层 → 深层）", fontsize=9.0, labelpad=3)
    ax.set_ylabel("输出标准差（对数轴）", fontsize=9.0, labelpad=4)
    ax.tick_params(labelsize=8.0, length=2.6, colors=fs.MUTED)
    fs.label(ax, 7.0, 6.5e-2, "灰带为 $0.5/\\sqrt{d}$ 到 $2/\\sqrt{d}$", fs=7.8,
             color=fs.MUTED, ha="center")
    fs.label(ax, 7.6, 4.2e-3, "越深越薄", fs=8.2, color=fs.ACCENT, ha="center")

    # (b) 有效秩归一化
    ax = axes[1]
    for r, ls in (("healthy", "-"), ("dim", (0, (5, 2.5)))):
        y = np.array([_eff(_path(r, t)) for t in ep / 100.0]) / d
        ax.plot(ep, y, color=fs.BLUE if r == "healthy" else fs.ACCENT, lw=1.8, ls=ls, zorder=4)
    ax.axhspan(0.0, 0.10, color=fs.GRID, alpha=0.75, zorder=1, lw=0)
    ax.axhline(0.10, color=fs.MUTED, lw=0.9, ls=(0, (1.5, 2.2)), zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.60)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.2, 0.4])
    fs.clean(ax, grid=True)
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("有效秩 / d", fontsize=9.0, labelpad=4)
    ax.tick_params(labelsize=8.0, length=2.6, colors=fs.MUTED)
    fs.label(ax, 97, 0.445, "健康 0.37", fs=8.2, color=fs.BLUE, ha="right")
    fs.label(ax, 20, 0.130, "报警线 0.10", fs=7.8, color=fs.MUTED, ha="left")
    ax.annotate("塌缩 0.03", xy=(58, 0.038), xytext=(76, 0.195),
                fontsize=8.2, color=fs.ACCENT, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=fs.ACCENT, lw=0.8,
                                shrinkA=3, shrinkB=2))

    # (c) 尾部能量占比
    ax = axes[2]
    for r, ls in (("healthy", "-"), ("dim", (0, (5, 2.5)))):
        y = np.array([_tail_share(_path(r, t)) for t in ep / 100.0]) * 100
        ax.semilogy(ep, np.maximum(y, 1e-8), color=fs.BLUE if r == "healthy" else fs.ACCENT,
                    lw=1.8, ls=ls, zorder=4)
    ax.axhline(49.0, color=fs.GREEN, lw=1.2, ls=(0, (1.5, 2.2)), zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(1e-7, 3e3)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1, 1e2])
    fs.clean(ax, grid=True)
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("后一半维度的能量占比（%）", fontsize=9.0, labelpad=4)
    ax.tick_params(labelsize=8.0, length=2.6, colors=fs.MUTED)
    fs.label(ax, 97, 300, "完全塌缩 49%", fs=8.2, color=fs.GREEN, ha="right")
    fs.label(ax, 97, 15.0, "健康 4.8%", fs=8.2, color=fs.BLUE, ha="right")
    fs.label(ax, 50, 8.0e-4, r"塌缩 $10^{-6}$% 量级", fs=8.2, color=fs.ACCENT, ha="center")

    # (d) 线性探针
    ax = axes[3]
    probe_ok = 62.0 * (1 - np.exp(-ep / 26.0))
    probe_bad = 0.6 + 0.4 * np.exp(-ep / 6.0)
    ax.plot(ep, probe_ok, color=fs.BLUE, lw=1.8, zorder=4)
    ax.plot(ep, probe_bad, color=fs.ACCENT, lw=1.8, ls=(0, (5, 2.5)), zorder=4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 74)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 20, 40, 60])
    fs.clean(ax, grid=True)
    ax.set_xlabel("轮次", fontsize=9.0, labelpad=3)
    ax.set_ylabel("线性探针准确率（%）", fontsize=9.0, labelpad=4)
    ax.tick_params(labelsize=8.0, length=2.6, colors=fs.MUTED)
    fs.label(ax, 30, 12.0, "千分之一，等于随机", fs=8.2, color=fs.ACCENT, ha="center")

    tags = ["(a) 逐层标准差", "(b) 有效秩", "(c) 尾部能量", "(d) 线性探针"]
    for ax, tg in zip(axes, tags):
        ax.set_title(tg, fontsize=8.9, color=fs.INK, pad=5, loc="left")

    return fs.save(fig, "practice", "ch02", "monitor-panel")


if __name__ == "__main__":
    fig_states()
    fig_spectrum()
    fig_monitors()
    report_numbers()
    fig_rank_curve()
    fig_monitor_panel()
