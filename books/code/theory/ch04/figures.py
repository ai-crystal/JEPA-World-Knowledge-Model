# -*- coding: utf-8 -*-
"""第 4 章插图：（1）定理 2 的数值验证；（2）两个核的谱阶梯与间隙的不对称。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


# ---------------------------------------------------------------- 数值工具
def sinkhorn(M, p, itmax=60000, tol=1e-13):
    """对称缩放：把对称非负矩阵 M 变成行和为 p 的对称矩阵。"""
    a = np.ones(len(p))
    Mo = M.copy()
    np.fill_diagonal(Mo, 0.0)
    Mo = Mo / Mo.sum()
    for _ in range(itmax):
        R = Mo * a[:, None] * a[None, :]
        row = R.sum(1)
        a = a * np.sqrt(p / row)
        if np.abs(row - p).max() < tol:
            break
    return Mo * a[:, None] * a[None, :]


def random_kernel(n, p, rng, clustered):
    if clustered:
        c = rng.integers(0, 2, n)
        M = np.where(c[:, None] == c[None, :],
                     rng.uniform(0.6, 1.0, (n, n)),
                     rng.uniform(0.02, 0.06, (n, n)))
    else:
        M = rng.uniform(0.0, 1.0, (n, n))
    M = (M + M.T) / 2
    return sinkhorn(M, p)


def spectrum(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def procrustes_error(A, B):
    """min over Q in O(k) of ||A - B Q^T||_F。"""
    U, s, Vt = np.linalg.svd(A.T @ B)
    Q = U @ Vt
    return np.linalg.norm(A - B @ Q.T), Q


# ---------------------------------------------------------------- 图 4-1
def theorem2_trials(count=80, seed=1):
    """随机有限世界：每个世界一对核，第二个核由第一个与一个无关核混合而成。

    返回每行的 (误差, 上界, 是否聚簇, k, 状态数, K2 的谱间隙, 子空间距离)。
    """
    rng = np.random.default_rng(seed)
    rows = []
    tries = 0
    while len(rows) < count and tries < 20000:
        tries += 1
        n = int(rng.integers(8, 40))
        k = min(int(rng.integers(1, 6)), n - 3)
        p = rng.uniform(0.3, 1.0, n)
        p = p / p.sum()
        clustered = bool(rng.random() < 0.5)
        P1 = random_kernel(n, p, rng, clustered)
        Pb = random_kernel(n, p, rng, False)
        s = np.sqrt(np.outer(p, p))
        K1 = P1 / s
        Kb = Pb / s
        if abs(np.linalg.eigvalsh(K1)[-1] - 1.0) > 1e-8:
            continue
        mix = 10 ** rng.uniform(-4, 0)
        K2 = (1 - mix) * K1 + mix * Kb
        w1, V1 = spectrum(K1)
        w2, V2 = spectrum(K2)
        delta = w2[k] - w2[k + 1]
        if delta <= 1e-6:
            continue
        A = V1[:, 1:k + 1]
        B = V2[:, 1:k + 1]
        err, _ = procrustes_error(A, B)
        diff = K1 - K2
        bound = (2 * np.sqrt(2) / delta) * min(np.sqrt(k) * np.linalg.norm(diff, 2),
                                               np.linalg.norm(diff, "fro"))
        if not np.isfinite(bound) or bound < 1e-10:
            continue
        sub = np.linalg.norm(A @ A.T - B @ B.T, "fro")
        rows.append((err, bound, clustered, k, n, delta, sub))
    return rows


def fig_theorem2():
    rows = theorem2_trials(80)
    err = np.array([r[0] for r in rows])
    bnd = np.array([r[1] for r in rows])
    clus = np.array([r[2] for r in rows])
    delta = np.array([r[5] for r in rows])
    kk = np.array([r[3] for r in rows])
    ratio = err / bnd
    print("  图 4-1  世界数", len(err),
          " 误差/上界 最大", round(ratio.max(), 4), " 中位", round(np.median(ratio), 4),
          " 误差范围", (round(err.min(), 6), round(err.max(), 3)),
          " 上界范围", (round(bnd.min(), 5), round(bnd.max(), 1)),
          " k 取值", sorted(set(kk.tolist())))
    sub = np.array([r[6] for r in rows])
    plain = bnd / (2 * np.sqrt(2))
    print("         误差/子空间距离 中位", round(np.median(err / sub), 4),
          " 子空间距离除以「核失配除以间隙」中位", round(np.median(sub / plain), 4))
    # 间隙最小三分之一与最大三分之一
    lo = ratio[delta < np.percentile(delta, 33)]
    hi = ratio[delta > np.percentile(delta, 67)]
    print("         小间隙组 误差/上界 中位", round(np.median(lo), 4),
          " 大间隙组 中位", round(np.median(hi), 4))

    fig, ax = plt.subplots(figsize=(6.5, 3.75))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.955, bottom=0.155)
    lo_, hi_ = 5e-5, 4e3
    ax.plot([lo_, hi_], [lo_, hi_], color=fs.INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.scatter(bnd[clus], err[clus], s=26, facecolor="white", edgecolor=fs.GREEN,
               linewidth=0.9, zorder=3, label="聚簇的世界（谱间隙小）")
    ax.scatter(bnd[~clus], err[~clus], s=24, color=fs.BLUE, zorder=3,
               edgecolor="white", linewidth=0.5, label="随机的世界")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_, hi_)
    ax.set_ylim(1e-5, 5.0)
    ax.set_xlabel("定理 2 上界，核失配除以谱间隙（对数刻度）", fontsize=9.2, labelpad=5)
    ax.set_ylabel("最优旋转后的误差（对数刻度）", fontsize=9.2, labelpad=4)
    fs.clean(ax, grid=True)
    ax.legend(loc="lower right", handletextpad=0.5, borderaxespad=0.9)
    ax.text(1.4e-4, 3.2e0, "每个点是一个随机的有限世界", fontsize=8.4, color=fs.MUTED,
            ha="left", va="top")
    # 对角线在屏幕上的倾角：对数坐标下斜率 1
    bb = ax.get_position()
    w_in = bb.width * fig.get_size_inches()[0]
    h_in = bb.height * fig.get_size_inches()[1]
    xd = np.log10(hi_) - np.log10(lo_)
    yd = np.log10(5.0) - np.log10(1e-5)
    ang = np.degrees(np.arctan((h_in / yd) / (w_in / xd)))
    ax.text(1.35e-3, 2.6e-3, "误差 = 上界", fontsize=8.4, color=fs.INK,
            ha="left", va="bottom", rotation=ang, rotation_mode="anchor")
    return fs.save(fig, "theory", "ch04", "theorem2-check")


# ---------------------------------------------------------------- 图 4-2
def pair_of_kernels(seed=5, n=24, mix=0.40):
    """一个可复现的世界：K1 是聚簇核，K2 混进了一部分无关核。"""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.3, 1.0, n)
    p = p / p.sum()
    P1 = random_kernel(n, p, rng, True)
    Pb = random_kernel(n, p, rng, False)
    s = np.sqrt(np.outer(p, p))
    K1 = P1 / s
    Kb = Pb / s
    w1, V1 = spectrum(K1)
    K2 = (1 - mix) * K1 + mix * Kb
    w2, V2 = spectrum(K2)
    return K1, K2, w1, V1, w2, V2, mix


def fig_gap_stability():
    K1, K2, w1, V1, w2, V2, mix = pair_of_kernels()
    g1 = w1[1] - w1[2]
    g2 = w2[1] - w2[2]
    print("  图 4-2  K1 谱", np.round(w1[:5], 4), " K2 谱", np.round(w2[:5], 4),
          " 混合权重", mix, " 间隙 K1 %.4f  K2 %.4f" % (g1, g2))

    fig = plt.figure(figsize=(7.0, 3.30))
    axa = fig.add_axes([0.085, 0.180, 0.385, 0.745])
    axb = fig.add_axes([0.595, 0.180, 0.370, 0.745])

    # ---------------- (a) 两个核的谱阶梯
    idx = np.arange(0, 5)
    for x, v in zip(idx, w1[:5]):
        axa.plot([x, x], [0, v], color=fs.BLUE, lw=1.0, alpha=0.85, zorder=2)
    for x, v in zip(idx, w2[:5]):
        axa.plot([x + 0.10, x + 0.10], [0, v], color=fs.ACCENT, lw=1.4, alpha=0.9, zorder=2)
    axa.plot(idx, w1[:5], marker="o", ms=6.0, mfc="white", mec=fs.BLUE, mew=1.1,
             ls="none", zorder=4)
    axa.plot(idx + 0.10, w2[:5], marker="s", ms=5.2, color=fs.ACCENT, ls="none", zorder=4)
    axa.axhline(0.0, color=fs.FAINT, lw=0.7, zorder=1)

    # 两个间隙并排画在 mu_1 与 mu_2 之间
    axa.annotate("", xy=(1.40, w1[1]), xytext=(1.40, w1[2]),
                 arrowprops=dict(arrowstyle="<->", color=fs.BLUE, lw=1.0,
                                 shrinkA=0, shrinkB=0))
    axa.annotate("", xy=(1.72, w2[1]), xytext=(1.72, w2[2]),
                 arrowprops=dict(arrowstyle="<->", color=fs.ACCENT, lw=1.3,
                                 shrinkA=0, shrinkB=0))
    fs.label(axa, 1.40, 0.925, "$\\delta_1$", fs=9.0, color=fs.BLUE, va="bottom")
    fs.label(axa, 1.80, 0.545, "$\\delta_2$", fs=9.0, color=fs.ACCENT, ha="left",
             va="bottom")

    axa.set_xlim(-0.45, 4.55)
    axa.set_ylim(-0.09, 1.08)
    axa.set_xticks(idx + 0.05)
    axa.set_xticklabels(["$\\mu_0$", "$\\mu_1$", "$\\mu_2$", "$\\mu_3$", "$\\mu_4$"],
                        fontsize=8.6)
    axa.set_yticks([0.0, 0.4, 0.8, 1.0])
    axa.set_yticklabels(["0", "0.4", "0.8", "1"])
    axa.tick_params(axis="y", labelsize=7.8)
    axa.set_ylabel("特征值", fontsize=9.2, labelpad=3)
    fs.clean(axa)
    axa.plot([], [], marker="o", ms=5.8, mfc="white", mec=fs.BLUE, mew=1.1, ls="none",
             label="$K_1$")
    axa.plot([], [], marker="s", ms=5.0, color=fs.ACCENT, ls="none", label="$K_2$")
    axa.legend(loc="upper right", handletextpad=0.4, borderaxespad=0.1,
               fontsize=8.4, ncol=2, columnspacing=0.8, bbox_to_anchor=(1.02, 1.03))
    fs.label(axa, 2.95, 0.225, "尾部三个特征值", fs=7.8, color=fs.MUTED)
    fs.label(axa, 2.95, 0.135, "挤在零附近", fs=7.8, color=fs.MUTED)
    axa.plot([1.94, 4.16], [0.055, 0.055], color=fs.GRID, lw=2.2,
             solid_capstyle="round", zorder=1)
    fs.label(axa, -0.38, 0.62, "(a)", fs=9.0, color=fs.MUTED, ha="left")

    # ---------------- (b) 间隙越小，上界离误差越远
    rows = theorem2_trials(80)
    err = np.array([r[0] for r in rows])
    bnd = np.array([r[1] for r in rows])
    clus = np.array([r[2] for r in rows])
    delta = np.array([r[5] for r in rows])
    ratio = err / bnd

    axb.scatter(delta[~clus], ratio[~clus], s=20, color=fs.BLUE, zorder=3,
                edgecolor="white", linewidth=0.4)
    axb.scatter(delta[clus], ratio[clus], s=22, facecolor="white", edgecolor=fs.GREEN,
                linewidth=0.9, zorder=3)
    lx = np.log(delta)
    ly = np.log(ratio)
    sl, ic = np.polyfit(lx, ly, 1)
    xm = float(np.exp(lx.mean()))
    ym = float(np.exp(ic + sl * lx.mean()))
    xx = np.array([delta.min() * 0.8, delta.max() * 1.2])
    axb.plot(xx, ym * (xx / xm) ** sl, color=fs.INK, lw=1.2, zorder=2)
    axb.plot(xx, ym * (xx / xm) ** 1.0, color=fs.FAINT, lw=1.0, ls=(0, (4, 3)), zorder=2)
    r = np.corrcoef(lx, ly)[0, 1]
    print("         间隙与误差/上界：斜率 %.3f 相关系数 %.3f" % (sl, r))

    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(8e-4, 2.0)
    axb.set_ylim(1.6e-3, 0.42)
    axb.set_xlabel("$K_2$ 的谱间隙 $\\delta$（对数刻度）", fontsize=9.2, labelpad=5)
    axb.set_ylabel("误差占上界的比例（对数刻度）", fontsize=9.2, labelpad=4)
    fs.clean(axb, grid=True)
    from matplotlib.lines import Line2D
    axb.legend(handles=[Line2D([], [], color=fs.INK, lw=1.2,
                              label="实测趋势，斜率 %.2f" % sl),
                        Line2D([], [], color=fs.FAINT, lw=1.0, ls=(0, (4, 3)),
                              label="参照：误差不随间隙变化")],
               loc="upper left", fontsize=7.8, handlelength=1.8, labelspacing=0.42,
               borderaxespad=0.25, handletextpad=0.6)
    fs.label(axb, 1.9, 0.36, "(b)", fs=9.0, color=fs.MUTED, va="top", ha="right")
    return fs.save(fig, "theory", "ch04", "gap-stability")


# ---------------------------------------------------------------- 图 4-3
# 主夹角与三种距离的关系。左侧是两组基按夹角配对，右侧把
# 「子空间距离」「投影算子距离」「基底距离」三条曲线画在一起，
# 说明常数里的 /2 是在哪一段上付掉的。
def fig_subspace_angle():
    from matplotlib.patches import Arc

    fig = plt.figure(figsize=(7.0, 3.30))
    axa = fig.add_axes([0.030, 0.120, 0.400, 0.820])
    axb = fig.add_axes([0.590, 0.175, 0.385, 0.745])

    # ---------------- (a) 两组基的配对与主夹角
    th1, th2 = np.deg2rad(22.0), np.deg2rad(20.0)
    O = np.array([0.0, 0.0])
    u1, u2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    v1 = np.array([np.cos(th1), np.sin(th1)])
    v2 = np.array([np.sin(th2), np.cos(th2)])

    for p1, col in ((u1, fs.BLUE), (u2, fs.BLUE), (v1, fs.ACCENT), (v2, fs.ACCENT)):
        fs.arrow(axa, O, p1, color=col, lw=1.5, ms=9)
    for p1, p2 in ((u1, v1), (u2, v2)):
        axa.plot([p1[0], p2[0]], [p1[1], p2[1]], color=fs.FAINT, lw=0.9,
                 ls=(0, (3, 2)), zorder=1)
    axa.add_patch(Arc(O, 1.02, 1.02, theta1=0.0, theta2=22.0, color=fs.MUTED, lw=1.0, zorder=2))
    axa.add_patch(Arc(O, 0.70, 0.70, theta1=70.0, theta2=90.0, color=fs.MUTED, lw=1.0,
                      zorder=2))
    fs.label(axa, 0.59, 0.115, r"$\theta_1$", fs=10.0, color=fs.INK)
    fs.label(axa, 0.075, 0.455, r"$\theta_2$", fs=10.0, color=fs.INK)
    fs.label(axa, 1.075, -0.075, r"$u_1$", fs=10.0, color=fs.BLUE)
    fs.label(axa, -0.085, 1.055, r"$u_2$", fs=10.0, color=fs.BLUE)
    fs.label(axa, 1.055, 0.455, r"$v_1$", fs=10.0, color=fs.ACCENT)
    fs.label(axa, 0.395, 1.115, r"$v_2$", fs=10.0, color=fs.ACCENT)
    fs.label(axa, 0.44, 0.80, r"$\Vert u_i-v_i\Vert^2=2-2\cos\theta_i$", fs=8.4,
             color=fs.MUTED, ha="left")
    fs.label(axa, -0.12, -0.235, "(a)", fs=9.0, color=fs.MUTED, ha="left")
    axa.set_xlim(-0.30, 1.42)
    axa.set_ylim(-0.30, 1.30)
    axa.set_aspect("equal")
    axa.axis("off")

    # ---------------- (b) 三种距离随夹角的变化（k = 2，两个夹角相同）
    t = np.linspace(0.0, np.pi / 2, 400)
    sin_f = np.sqrt(2.0) * np.sin(t)          # ||sin Theta||_F
    proj = 2.0 * np.sin(t)                    # ||P1 - P2||_F = sqrt(2) * 上式
    basis = 2.0 * np.sqrt(1.0 - np.cos(t))    # ||V1 O - V2||_F = sqrt(4(1-cos))

    axb.fill_between(np.rad2deg(t), sin_f, proj, color=fs.BLUE, alpha=0.10, lw=0)
    axb.plot(np.rad2deg(t), proj, color=fs.FAINT, lw=1.3, ls=(0, (4, 3)), zorder=2)
    axb.plot(np.rad2deg(t), basis, color=fs.ACCENT, lw=1.8, zorder=3)
    axb.plot(np.rad2deg(t), sin_f, color=fs.BLUE, lw=1.5, zorder=4)
    fs.label(axb, 5.0, 2.62, r"阴影带就是 $\sqrt{2}$ 倍的余量", fs=8.4, color=fs.MUTED,
             ha="left", va="top")
    axb.plot([90.0], [2.0], marker="o", ms=4.6, mfc="white", mec=fs.INK, mew=1.0, zorder=6)
    axb.set_xlim(0.0, 96.0)
    axb.set_ylim(0.0, 2.70)
    axb.set_xticks([0, 30, 60, 90])
    axb.set_yticks([0.0, 1.0, 2.0])
    axb.set_xlabel("主夹角 $\\theta$（度）", fontsize=9.2, labelpad=5)
    axb.set_ylabel("距离", fontsize=9.2, labelpad=3)
    fs.clean(axb, grid=True)
    axb.plot([], [], color=fs.BLUE, lw=1.5, label=r"$\Vert\sin\Theta\Vert_F$")
    axb.plot([], [], color=fs.ACCENT, lw=1.8, label=r"$\Vert V_1\hat O-V_2\Vert_F$")
    axb.plot([], [], color=fs.FAINT, lw=1.3, ls=(0, (4, 3)),
             label=r"$\Vert P_1-P_2\Vert_F$")
    axb.legend(loc="lower right", fontsize=7.8, handlelength=1.9, labelspacing=0.35,
               borderaxespad=0.35, handletextpad=0.55)
    fs.label(axb, 95.0, 2.62, "(b)", fs=9.0, color=fs.MUTED, ha="right", va="top")
    print("  图 4-3  夹角-距离：theta=22° 时 basis %.4f  sinF %.4f  proj %.4f"
          % (2 * np.sqrt(1 - np.cos(np.deg2rad(22.0))),
             np.sqrt(2) * np.sin(np.deg2rad(22.0)), 2 * np.sin(np.deg2rad(22.0))))
    return fs.save(fig, "theory", "ch04", "subspace-angle")


# ---------------------------------------------------------------- 图 4-4
# 误差与谱间隙的散点，以及 Sinkhorn 缩放之后的边缘误差。
_MARGINAL_LOG = []


def _sinkhorn_trace(M, p, itmax=60000, tol=1e-13):
    """与 sinkhorn 同样的迭代，额外返回迭代次数与最终边缘误差。"""
    a = np.ones(len(p))
    Mo = M.copy()
    np.fill_diagonal(Mo, 0.0)
    Mo = Mo / Mo.sum()
    it = 0
    for it in range(itmax):
        R = Mo * a[:, None] * a[None, :]
        row = R.sum(1)
        a = a * np.sqrt(p / row)
        if np.abs(row - p).max() < tol:
            break
    R = Mo * a[:, None] * a[None, :]
    return R, it + 1, float(np.abs(R.sum(1) - p).max())


def _sinkhorn_recording(M, p, itmax=60000, tol=1e-13):
    """把边缘误差顺便记下来，供图 4-4(b) 使用。"""
    R, it, dev = _sinkhorn_trace(M, p, itmax, tol)
    _MARGINAL_LOG.append((it, dev))
    return R


def fig_error_gap():
    global sinkhorn
    _MARGINAL_LOG.clear()
    backup = sinkhorn
    sinkhorn = _sinkhorn_recording
    try:
        rows = theorem2_trials(80)
    finally:
        sinkhorn = backup

    err = np.array([r[0] for r in rows])
    delta = np.array([r[5] for r in rows])
    clus = np.array([r[2] for r in rows])
    its = np.array([m[0] for m in _MARGINAL_LOG], float)
    dev = np.array([m[1] for m in _MARGINAL_LOG], float)
    lx, ly = np.log(delta), np.log(err)
    sl, ic = np.polyfit(lx, ly, 1)
    print("  图 4-4  误差与间隙：斜率 %.3f 相关 %.3f；Sinkhorn 迭代 %d-%d 中位 %d；"
          " 边缘误差 中位 %.2e 最大 %.2e"
          % (sl, np.corrcoef(lx, ly)[0, 1], its.min(), its.max(), np.median(its),
             np.median(dev), dev.max()))

    fig = plt.figure(figsize=(7.0, 3.30))
    axa = fig.add_axes([0.090, 0.180, 0.385, 0.745])
    axb = fig.add_axes([0.595, 0.180, 0.380, 0.745])

    # ---------------- (a) 误差与间隙
    axa.scatter(delta[~clus], err[~clus], s=22, color=fs.BLUE, zorder=3,
                edgecolor="white", linewidth=0.4)
    axa.scatter(delta[clus], err[clus], s=24, facecolor="white", edgecolor=fs.GREEN,
                linewidth=0.9, zorder=3)
    xx = np.array([8e-4, 2.0])
    xm = float(np.exp(lx.mean()))
    ym = float(np.exp(ic + sl * lx.mean()))
    axa.plot(xx, ym * (xx / xm) ** sl, color=fs.INK, lw=1.2, zorder=2)
    axa.plot(xx, ym * (xx / xm) ** (-1.0), color=fs.FAINT, lw=1.0, ls=(0, (4, 3)), zorder=2)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(8e-4, 2.0)
    axa.set_ylim(1e-5, 5.0)
    axa.set_xlabel("$K_2$ 的谱间隙 $\\delta$（对数刻度）", fontsize=9.2, labelpad=5)
    axa.set_ylabel("最优旋转后的误差（对数刻度）", fontsize=9.2, labelpad=4)
    fs.clean(axa, grid=True)
    from matplotlib.lines import Line2D
    axa.legend(handles=[Line2D([], [], color=fs.INK, lw=1.2,
                               label="拟合斜率 %.2f" % sl),
                        Line2D([], [], color=fs.FAINT, lw=1.0, ls=(0, (4, 3)),
                               label="参照：误差按 $1/\\delta$")],
               loc="upper right", fontsize=7.6, handlelength=1.8, labelspacing=0.40,
               borderaxespad=0.25, handletextpad=0.55)
    fs.label(axa, 0.015, 0.035, "(a)", fs=9.0, color=fs.MUTED, ha="left", va="bottom",
             transform=axa.transAxes)

    # ---------------- (b) Sinkhorn 之后的边缘误差
    idx = np.arange(1, len(dev) + 1)
    axb.plot(idx, dev, marker="o", ms=2.6, ls="none", color=fs.BLUE, alpha=0.85, zorder=3)
    axb.axhline(1e-13, color=fs.ACCENT, lw=1.0, ls=(0, (4, 3)), zorder=2)
    axb.set_yscale("log")
    axb.set_xlim(0, 84)
    axb.set_ylim(1e-16, 1e-11)
    axb.set_xlabel("世界序号", fontsize=9.2, labelpad=5)
    axb.set_ylabel("边缘误差（对数刻度）", fontsize=9.2, labelpad=4)
    axb.set_yticks([1e-16, 1e-14, 1e-12])
    fs.clean(axb, grid=True)
    fs.label(axb, 82.0, 1.35e-13, "停机容差 $10^{-13}$", fs=7.8, color=fs.ACCENT,
             ha="right", va="bottom")
    fs.label(axb, 82.0, 2.2e-14, "中位 $%.1f\\times10^{-14}$" % (np.median(dev) * 1e14),
             fs=7.8, color=fs.MUTED, ha="right", va="top")
    fs.label(axb, 0.03, 0.955, "(b)", fs=9.0, color=fs.MUTED, ha="left", va="top",
             transform=axb.transAxes)
    return fs.save(fig, "theory", "ch04", "error-gap")


if __name__ == "__main__":
    fig_theorem2()
    fig_gap_stability()
    fig_subspace_angle()
    fig_error_gap()
