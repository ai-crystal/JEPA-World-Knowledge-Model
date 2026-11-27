# -*- coding: utf-8 -*-
"""第 9 章插图：谱方法的四类误差来源（有限样本、间隙偏差、符号与旋转、数值迭代）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon
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


def normalized_adjacency(Pi, p):
    return Pi / np.sqrt(np.outer(p, p))


def spectrum(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def build_toy_world():
    """12 个状态排成 4x3 的格子，正样本偏好相邻状态（与第 2 章同一个世界）。"""
    nx, ny = 4, 3
    n = nx * ny
    xy = np.array([(i, j) for j in range(ny) for i in range(nx)], float)
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    M = np.exp(-d2 / (2 * 0.9 ** 2))
    rng = np.random.default_rng(3)
    p = np.full(n, 1.0 / n) + rng.uniform(-0.03, 0.03, n) / n
    p = p / p.sum()
    Pi = sinkhorn(M, p)
    return xy, p, Pi


def build_dense_world(n=40, seed=7):
    """40 个状态、连接稠密的世界：谱头几个挤在一起，真间隙很小。"""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1, (n, 2))
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    M = np.exp(-d2 / (2 * 0.55 ** 2)) * 0.35 + rng.uniform(0.30, 1.0, (n, n))
    M = (M + M.T) / 2
    p = np.full(n, 1.0 / n)
    return sinkhorn(M, p), p


def sample_pairs(Pi, N, rng):
    """按 pi 抽 N 对正样本，返回两头的状态编号。"""
    n = Pi.shape[0]
    flat = Pi.ravel() / Pi.sum()
    idx = rng.choice(n * n, size=N, p=flat)
    return idx // n, idx % n


def empirical_kernel(Pi, N, rng):
    """配对抽样给出的经验核：返回 (Pi_hat, p_hat)。两头都记账，边缘自动相同。"""
    n = Pi.shape[0]
    z, zz = sample_pairs(Pi, N, rng)
    H = np.zeros((n, n))
    np.add.at(H, (z, zz), 1.0)
    np.add.at(H, (zz, z), 1.0)
    H /= 2.0 * N
    return H, H.sum(1)


def bernstein_t(N, d, eta=0.05):
    """矩阵 Bernstein：P(||Pi_hat-Pi||_2 >= t) <= 2d exp(-N t^2 / (2(1+t/3)))。解出 t。"""
    f = lambda t: 2.0 * d * np.exp(-N * t ** 2 / (2.0 * (1.0 + t / 3.0))) - eta
    lo, hi = 1e-9, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- 图 9-1
DELTA = 0.2581
NS = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
               768, 1024, 1536, 2048, 3072, 4096, 6144, 8192])


def sample_error_curve(trials=140, seed=0):
    xy, p, Pi = build_toy_world()
    n = len(p)
    K = normalized_adjacency(Pi, p)
    rng = np.random.default_rng(seed)
    e2, eF, k2, kF, miss = [], [], [], [], []
    for N in NS:
        a2, aF, b2, bF, m = [], [], [], [], 0
        for _ in range(trials):
            H, ph = empirical_kernel(Pi, N, rng)
            if ph.min() <= 0:
                m += 1
                continue
            Kh = H / np.sqrt(np.outer(ph, ph))
            dP = H - Pi
            dK = Kh - K
            a2.append(np.linalg.norm(dP, 2))
            aF.append(np.linalg.norm(dP, "fro"))
            b2.append(np.linalg.norm(dK, 2))
            bF.append(np.linalg.norm(dK, "fro"))
        e2.append(np.median(a2)); eF.append(np.median(aF))
        k2.append(np.median(b2)); kF.append(np.median(bF))
        miss.append(m / trials)
    return (np.array(e2), np.array(eF), np.array(k2), np.array(kF), np.array(miss), n)


def fig_sample_error():
    e2, eF, k2, kF, miss, n = sample_error_curve()
    bound = np.array([bernstein_t(N, n) for N in NS])
    print("  图 9-1  状态数", n)
    for N, a, b, c, d, t in zip(NS, e2, eF, k2, kF, bound):
        if N in (16, 32, 128, 512, 2048, 8192):
            print("         N=%5d  ||Pi-Pi^||_2 %.4f  ||Pi-Pi^||_F %.4f  "
                  "||K-K^||_2 %.4f  ||K-K^||_F %.4f  Bernstein %.4f  放大 %.2f"
                  % (N, a, b, c, d, t, c / a))
    print("         未抽全的试验比例（N=16,24,32）", np.round(miss[:3], 3))
    print("         实测/上界 中位", round(float(np.median(e2 / bound)), 4))
    # 上界降到 delta/10 所需的 N
    delta = 0.2580
    ok = [N for N in range(16, 400000) if bernstein_t(N, n) <= delta / 10]
    print("         Bernstein 上界降到 delta/10 的 N =", ok[0] if ok else None)
    idx = np.searchsorted(NS, NS[e2 <= delta / 10][0]) if np.any(e2 <= delta / 10) else None
    print("         实测 2-范数降到 delta/10 的 N =", NS[idx] if idx is not None else None)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 3.25))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.945, bottom=0.185, wspace=0.31)

    # ---- (a) 未归一化的经验核
    axa.loglog(NS, e2, color=fs.BLUE, lw=1.5, marker="o", ms=3.6, mfc="white",
               mew=0.9, zorder=4, label=r"实测 $\|\hat\Pi-\Pi\|_2$")
    axa.loglog(NS, eF, color=fs.GREEN, lw=1.3, marker="^", ms=3.4, mfc="white",
               mew=0.8, zorder=4, label=r"实测 $\|\hat\Pi-\Pi\|_F$")
    axa.loglog(NS, bound, color=fs.INK, lw=1.2, ls=(0, (4, 3)), zorder=5,
               label=r"Bernstein 上界（$\eta=0.05$）")
    axa.axvline(n, color=fs.FAINT, lw=0.9, ls=":", zorder=1)
    fs.label(axa, n * 1.18, 2.1, r"$N=|Z|$", fs=8.4, color=fs.MUTED, ha="left", va="top")
    axa.set_xlabel("正样本对数 $N$（对数刻度）", fontsize=9.0, labelpad=4)
    axa.set_ylabel("未归一化核的误差（对数刻度）", fontsize=9.0, labelpad=4)
    axa.set_ylim(2.9e-4, 2.6)
    axa.set_xlim(9.0, 1.6e4)
    fs.clean(axa, grid=True)
    axa.legend(loc="lower left", handletextpad=0.6, borderaxespad=0.5, labelspacing=0.42,
               fontsize=8.6)
    axa.text(0.965, 0.955, "(a)", transform=axa.transAxes, fontsize=9.4, color=fs.INK,
             ha="right", va="top", weight="bold")

    # ---- (b) 归一化之后，与间隙比
    axb.loglog(NS, k2, color=fs.ACCENT, lw=1.5, marker="s", ms=3.4, zorder=4,
               label=r"实测 $\|\hat K-K\|_2$")
    axb.loglog(NS, kF, color=fs.PURPLE, lw=1.3, marker="D", ms=3.0, mfc="white",
               mew=0.8, zorder=4, label=r"实测 $\|\hat K-K\|_F$")
    axb.axhline(DELTA, color=fs.INK, lw=1.0, ls=(0, (4, 3)), zorder=3)
    fs.label(axb, 1.35e4, DELTA * 1.16, r"谱间隙 $\delta=%.3f$" % DELTA, fs=8.4,
             color=fs.INK, ha="right", va="bottom")
    axb.set_xlabel("正样本对数 $N$（对数刻度）", fontsize=9.0, labelpad=4)
    axb.set_ylabel("归一化核的误差（对数刻度）", fontsize=9.0, labelpad=4)
    axb.set_ylim(6e-3, 3.0)
    axb.set_xlim(9.0, 1.6e4)
    fs.clean(axb, grid=True)
    axb.legend(loc="lower left", handletextpad=0.6, borderaxespad=0.5, labelspacing=0.42,
               fontsize=8.6)
    axb.text(0.965, 0.955, "(b)", transform=axb.transAxes, fontsize=9.4, color=fs.INK,
             ha="right", va="top", weight="bold")
    return fs.save(fig, "theory", "ch09", "sample-error")


# ---------------------------------------------------------------- 图 9-2
def gap_bias_curve(Pi, p, k=1, Ns=NS, trials=160, seed=11):
    n = len(p)
    K = normalized_adjacency(Pi, p)
    w, V = spectrum(K)
    delta = w[k] - w[k + 1]
    rng = np.random.default_rng(seed)
    bias, ratio, nonpos, miss = [], [], [], []
    for N in Ns:
        b, r, z, gone = [], [], 0, 0
        for _ in range(trials):
            H, ph = empirical_kernel(Pi, N, rng)
            if ph.min() <= 0:
                gone += 1
                continue
            Kh = H / np.sqrt(np.outer(ph, ph))
            wh, _ = spectrum(Kh)
            dh = wh[k] - wh[k + 1]
            b.append((dh - delta) / delta)
            if dh <= 0:
                z += 1
                r.append(np.nan)
            else:
                r.append((1.0 / dh) / (1.0 / delta))
        bias.append(np.array(b) if b else np.array([np.nan]))
        ratio.append(np.array(r) if r else np.array([np.nan]))
        nonpos.append(z / max(1, len(b)))
        miss.append(gone / trials)
    return delta, bias, ratio, nonpos, np.array(miss)


def fig_gap_bias():
    xy, p, Pi = build_toy_world()
    Pi2, p2 = build_dense_world()
    dA, bA, rA, zA, mA = gap_bias_curve(Pi, p, 1)
    NB = np.array([128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
                   4096, 6144, 8192, 12288, 16384])
    dB, bB, rB, zB, mB = gap_bias_curve(Pi2, p2, 1, Ns=NB, seed=13, trials=400)
    print("  图 9-2  格子世界 delta = %.4f，稠密世界 delta = %.4f" % (dA, dB))
    print("         稠密世界谱", np.round(spectrum(normalized_adjacency(Pi2, p2))[0][:5], 4))
    for i, N in enumerate(NS):
        if N in (32, 128, 512, 2048, 8192):
            print("         N=%5d  相对偏差A %+.4f  1/delta 比值 A %.3f  未抽全 %.3f"
                  % (N, np.median(bA[i]), np.nanmedian(rA[i]), mA[i]))
    for i, N in enumerate(NB):
        if N in (128, 512, 2048, 8192, 16384):
            print("         N=%5d  相对偏差B %+.4f  1/delta 比值 B %.3f  未抽全 %.3f"
                  % (N, np.median(bB[i]), np.nanmedian(rB[i]), mB[i]))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 3.25))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.945, bottom=0.185, wspace=0.31)

    for Ns, b, r, c, lab in ((NS, bA, rA, fs.BLUE, "格子世界，$\\delta=%.3f$" % dA),
                             (NB, bB, rB, fs.ACCENT, "稠密世界，$\\delta=%.3f$" % dB)):
        med = np.array([np.nanmedian(v) for v in b])
        lo = np.array([np.nanpercentile(v, 25) for v in b])
        hi = np.array([np.nanpercentile(v, 75) for v in b])
        axa.fill_between(Ns, lo, hi, color=c, alpha=0.13, lw=0, zorder=2)
        axa.semilogx(Ns, med, color=c, lw=1.5, marker="o", ms=3.4, mfc="white",
                     mew=0.9, zorder=4, label=lab)
        rm = np.array([np.nanmedian(v) for v in r])
        axb.semilogx(Ns, rm, color=c, lw=1.5, marker="o", ms=3.4, mfc="white",
                     mew=0.9, zorder=4, label=lab)

    axa.axhline(0.0, color=fs.INK, lw=0.9, ls=(0, (4, 3)), zorder=3)
    axa.set_xlabel("正样本对数 $N$（对数刻度）", fontsize=9.0, labelpad=4)
    axa.set_ylabel(r"间隙的相对偏差 $(\hat\delta-\delta)/\delta$", fontsize=9.0, labelpad=4)
    axa.set_ylim(-0.95, 4.55)
    axa.set_xlim(12, 2.2e4)
    fs.clean(axa, grid=True)
    axa.legend(loc="upper right", handletextpad=0.6, borderaxespad=0.7, labelspacing=0.42,
               fontsize=8.5)
    axa.text(0.03, 0.955, "(a)", transform=axa.transAxes, fontsize=9.4, color=fs.INK,
             ha="left", va="top", weight="bold")
    axa.text(13.5, 2.55, "间隙被撑开", fontsize=8.4, color=fs.MUTED, ha="left", va="bottom")
    axa.text(13.5, -0.66, "间隙被抹平", fontsize=8.4, color=fs.MUTED, ha="left", va="top")

    axb.axhline(1.0, color=fs.INK, lw=0.9, ls=(0, (4, 3)), zorder=3)
    axb.set_xlabel("正样本对数 $N$（对数刻度）", fontsize=9.0, labelpad=4)
    axb.set_ylabel(r"插值得到的 $1/\hat\delta$ 相对真值", fontsize=9.0, labelpad=4)
    axb.set_ylim(0.0, 1.66)
    axb.set_xlim(12, 2.6e4)
    fs.clean(axb, grid=True)
    axb.text(0.965, 0.955, "(b)", transform=axb.transAxes, fontsize=9.4, color=fs.INK,
             ha="right", va="top", weight="bold")
    axb.text(4.2e3, 1.10, "格子世界", fontsize=8.6, color=fs.BLUE, ha="left", va="bottom")
    axb.text(4.2e3, 0.60, "稠密世界", fontsize=8.6, color=fs.ACCENT, ha="left", va="top")
    axb.text(0.30, 0.86, "偏保守", transform=axb.transAxes, fontsize=8.4, color=fs.MUTED,
             ha="left", va="bottom")
    axb.text(0.22, 0.24, "偏乐观", transform=axb.transAxes, fontsize=8.4, color=fs.MUTED,
             ha="left", va="top")
    return fs.save(fig, "theory", "ch09", "gap-bias")


# ---------------------------------------------------------------- 图 9-3
def build_random_world(n=12, seed=4):
    """一个没有对称性的 12 状态世界：随机摆点，距离近的更容易配成正样本。"""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 1.0, (n, 2))
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    M = np.exp(-d2 / (2 * 0.30 ** 2))
    p = np.full(n, 1.0 / n)
    return sinkhorn(M, p), p


def fig_sign_flip():
    Pi, p = build_random_world()
    n = len(p)
    K = normalized_adjacency(Pi, p)
    w, V = spectrum(K)
    Phi = V[:, 1:3] / np.sqrt(p)[:, None]

    # 第二列取反：一个正交变换，距离不变
    Phif = Phi * np.array([1.0, -1.0])
    d0 = np.linalg.norm(Phi[:, None, :] - Phi[None, :, :], axis=-1)
    d1 = np.linalg.norm(Phif[:, None, :] - Phif[None, :, :], axis=-1)
    print("  图 9-3  第二列取反后，两两距离最大改变 %.2e" % np.abs(d0 - d1).max())
    print("         两列绝对值最大处的符号", (V[:, 1][np.argmax(np.abs(V[:, 1]))] > 0,
                                          V[:, 2][np.argmax(np.abs(V[:, 2]))] > 0))
    print("         该世界 top-2 谱", np.round(w[:4], 4))

    # 简并块：二重特征值 + 块内扰动使基旋转 45 度，平面不动
    lam = np.array([1.0, 0.60, 0.60, 0.20, 0.05])
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    A = Q @ np.diag(lam) @ Q.T
    A = (A + A.T) / 2
    wa, Va = spectrum(A)
    u2, u3 = Va[:, 1], Va[:, 2]
    B = A + 0.012 * (np.outer(u2, u3) + np.outer(u3, u2))
    wb, Vb = spectrum(B)
    cosv = np.abs(Va[:, 1:3].T @ Vb[:, 1:3])
    phi = np.degrees(np.arctan2(cosv[0, 1], cosv[0, 0]))
    Pa = Va[:, 1:3] @ Va[:, 1:3].T
    Pb = Vb[:, 1:3] @ Vb[:, 1:3].T
    print("         简并块：基向量旋转 %.1f 度，子空间距离 %.2e，新特征值 %s"
          % (phi, np.linalg.norm(Pa - Pb, "fro"), np.round(wb[1:3], 4)))
    # 块外扰动才动平面：加一个块外分量再看
    B2 = B + 0.004 * (np.outer(Va[:, 2], Va[:, 3]) + np.outer(Va[:, 3], Va[:, 2]))
    _, Vb2 = spectrum(B2)
    print("         加块外扰动 0.004 后，子空间距离 %.2e"
          % np.linalg.norm(Pa - Vb2[:, 1:3] @ Vb2[:, 1:3].T, "fro"))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 3.20))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.06, wspace=0.10)

    # ---- (a) 同一次分解的两套坐标，只差第二列的符号
    axa.axis("off")
    axa.set_xlim(-1.95, 2.15); axa.set_ylim(-1.95, 2.05)
    axa.set_aspect("equal")
    axa.annotate("", xy=(2.02, 0), xytext=(-1.80, 0),
                 arrowprops=dict(arrowstyle="-|>", color=fs.FAINT, lw=0.9))
    axa.annotate("", xy=(0, 1.92), xytext=(0, -1.80),
                 arrowprops=dict(arrowstyle="-|>", color=fs.FAINT, lw=0.9))
    for i in range(n):
        axa.plot([Phi[i, 0], Phif[i, 0]], [Phi[i, 1], Phif[i, 1]],
                 color="#DCDCDC", lw=0.7, zorder=1)
    axa.scatter(Phi[:, 0], Phi[:, 1], s=26, color=fs.BLUE, zorder=4,
                edgecolor="white", linewidth=0.5, label="一次特征分解")
    axa.scatter(Phif[:, 0], Phif[:, 1], s=62, facecolor="none", edgecolor=fs.ACCENT,
                linewidth=1.1, zorder=3, label="第二列取相反符号")
    fs.label(axa, 1.78, 0.10, r"$\phi_1$", fs=9.0, color=fs.MUTED, ha="left")
    fs.label(axa, 0.09, 1.74, r"$\phi_2$", fs=9.0, color=fs.MUTED, ha="left")
    axa.text(0.02, 0.99, "(a)", transform=axa.transAxes, fontsize=9.4, ha="left",
             va="top", weight="bold")
    axa.legend(loc="lower left", handletextpad=0.5, borderaxespad=0.1,
               bbox_to_anchor=(-0.03, -0.04), labelspacing=0.35, fontsize=8.5)

    # ---- (b) 简并块：基转 45 度，平面不动
    axb.axis("off")
    axb.set_xlim(-1.60, 1.60); axb.set_ylim(-1.50, 1.42)
    axb.set_aspect("equal")
    axb.add_patch(Polygon([(-1.22, -1.10), (1.22, -1.10), (1.22, 1.16), (-1.22, 1.16)],
                          closed=True, facecolor="#F1F0EC", edgecolor="none", zorder=0))
    axb.add_patch(Arc((0, 0), 0.85, 0.85, theta1=0, theta2=45, color=fs.GOLD, lw=1.1, zorder=3))
    fs.label(axb, 0.53, 0.24, r"$45^\circ$", fs=8.6, color=fs.GOLD, ha="left")
    for vx, vy in ((1, 0), (0, 1)):
        axb.annotate("", xy=(vx * 1.02, vy * 1.02), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=1.3,
                                     shrinkA=0, shrinkB=0))
    c = np.cos(np.pi / 4)
    for vx, vy in ((c, c), (-c, c)):
        axb.annotate("", xy=(vx * 1.02, vy * 1.02), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.3, ls=(0, (3, 2)),
                                     shrinkA=0, shrinkB=0))
    fs.label(axb, 1.05, -0.12, r"$v_2$", fs=9.0, color=fs.BLUE, ha="left")
    fs.label(axb, -0.06, 1.06, r"$v_3$", fs=9.0, color=fs.BLUE, ha="right")
    fs.label(axb, 0.80, 0.86, r"$\hat v_2$", fs=9.0, color=fs.ACCENT, ha="left")
    fs.label(axb, -0.80, 0.86, r"$\hat v_3$", fs=9.0, color=fs.ACCENT, ha="right")
    fs.label(axb, 0.0, -1.30, "同一个平面，两套基", fs=8.8, color=fs.MUTED)
    axb.text(0.02, 0.99, "(b)", transform=axb.transAxes, fontsize=9.4, ha="left",
             va="top", weight="bold")
    return fs.save(fig, "theory", "ch09", "sign-flip")


# ---------------------------------------------------------------- 图 9-4
def power_iteration_curves(iters=150):
    xy, p, Pi = build_toy_world()
    n = len(p)
    K = normalized_adjacency(Pi, p)
    w, V = spectrum(K)
    v1 = V[:, 1]
    P2 = V[:, 1:3] @ V[:, 1:3].T
    root = np.sqrt(p)[:, None]
    Kd = K - root @ root.T          # 去掉常数方向
    Ks = Kd + np.eye(n)             # 移位：把谱抬到非负

    def res_vec(u, v):
        """单向量与 v 的夹角正弦，用残差范数算，不靠 1-cos^2。"""
        u = u / np.linalg.norm(u)
        return np.linalg.norm(u - (u @ v) * v)

    rng = np.random.default_rng(21)
    u = rng.normal(size=n); u /= np.linalg.norm(u)
    e_plain, e_pow, e_shift, e_sub = [], [], [], []
    up, uq, us = u.copy(), u.copy(), u.copy()
    Qs, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    for t in range(iters + 1):
        e_plain.append(res_vec(up, v1))
        e_pow.append(res_vec(uq, v1))
        e_shift.append(res_vec(us, v1))
        e_sub.append(np.linalg.norm(Qs @ Qs.T - P2, 2))
        up = K @ up; up /= np.linalg.norm(up)
        uq = Kd @ uq; uq /= np.linalg.norm(uq)
        us = Ks @ us; us /= np.linalg.norm(us)
        Qs, _ = np.linalg.qr(Kd @ Qs)
    e_plain, e_pow, e_shift, e_sub = map(np.array, (e_plain, e_pow, e_shift, e_sub))
    # 幂迭代的真实瓶颈：未知方向里模最大的那个特征值，负端可能赢过 mu_2
    tail = np.abs(w[2:])
    r_start = tail.max() / w[1]
    r_shift = np.abs(1.0 + w[2:]).max() / (1.0 + w[1])
    hit = lambda e: (int(np.argmax(e < 1e-10)) if np.any(e < 1e-10) else -1)
    t_pow, t_shift, t_sub = hit(e_pow), hit(e_shift), hit(e_sub)
    print("  图 9-4  mu_2/mu_1 = %.4f，但负端 |mu_11|/mu_1 = %.4f" %
          (w[2] / w[1], abs(w[-1]) / w[1]))
    print("         幂迭代达到 1e-10：%d 步（按 mu_2/mu_1 估算 %.1f，按负端 %.1f）"
          % (t_pow, np.log(1e-10) / np.log(w[2] / w[1]), np.log(1e-10) / np.log(r_start)))
    print("         移位后达到 1e-10：%d 步（估算 %.1f），斜率 %.4f"
          % (t_shift, np.log(1e-10) / np.log(r_shift), r_shift))
    print("         子空间迭代 k=2 的终点误差 %.4f（判断是否收敛：%s），实测斜率"
          % (e_sub[-1], t_sub > 0))
    ok = e_pow > 1e-6
    slope = np.polyfit(np.arange(len(e_pow))[ok][-30:],
                       np.log(e_pow[ok][-30:]), 1)[0]
    print("         实测斜率 exp(%.4f) = %.4f" % (slope, np.exp(slope)))
    print("         未去常数方向的终点误差 %.4f，sqrt(p) 与 v1 的内积 %.2e"
          % (e_plain[-1], float(np.sqrt(p) @ v1)))
    return (e_plain, e_pow, e_shift, e_sub, r_start, r_shift, t_pow, t_shift, w)


def fig_power_iter():
    (e_plain, e_pow, e_shift, e_sub, r_start, r_shift, t_pow, t_shift, w) = power_iteration_curves()
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.1, 3.30),
                                  gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.subplots_adjust(left=0.095, right=0.980, top=0.950, bottom=0.175, wspace=0.34)
    tt = np.arange(len(e_pow))

    ax.semilogy(tt, np.maximum(e_pow, 1e-15), color=fs.BLUE, lw=1.6, zorder=4,
                label=r"去常数方向的幂迭代，%d 步" % t_pow)
    ax.semilogy(tt, np.maximum(e_shift, 1e-15), color=fs.ACCENT, lw=1.6, zorder=4,
                label=r"移位之后再迭代 $K+I$，%d 步" % t_shift)
    ax.semilogy(tt, np.maximum(e_sub, 1e-15), color=fs.GREEN, lw=1.4, ls=(0, (5, 2)),
                zorder=4, label=r"子空间迭代 $k=2$")
    t0 = 3
    ax.semilogy(tt, np.maximum(e_pow[t0] * r_start ** (tt - t0), 1e-15), color=fs.INK,
                lw=0.8, ls=(0, (1.2, 2.2)), zorder=2)
    ax.semilogy(tt, np.maximum(e_shift[t0] * r_shift ** (tt - t0), 1e-15), color=fs.INK,
                lw=0.8, ls=(0, (1.2, 2.2)), zorder=2)
    fs.label(ax, 100, 4.0e-11, r"斜率 %.3f" % r_start, fs=8.4, color=fs.BLUE,
             ha="left", va="bottom")
    fs.label(ax, 118, 3e-7, r"斜率 %.3f" % r_shift, fs=8.4, color=fs.ACCENT,
             ha="left", va="bottom")
    for t_, e_, c_, s_ in ((t_pow, e_pow[t_pow], fs.BLUE, "o"),
                           (t_shift, e_shift[t_shift], fs.ACCENT, "s")):
        if t_ > 0:
            ax.plot([t_], [e_], marker=s_, ms=5.4, mfc="white", mec=c_, mew=1.3, zorder=6)
    ax.axhline(1.0, color=fs.FAINT, lw=0.8, zorder=1)
    fs.label(ax, 122, 1.25, "停在 1.00：收敛到别的子空间", fs=8.2, color=fs.GREEN,
             ha="right", va="bottom")
    ax.set_xlabel("迭代次数", fontsize=9.0, labelpad=4)
    ax.set_ylabel("与目标方向的夹角正弦（对数刻度）", fontsize=9.0, labelpad=4)
    ax.set_xlim(0, len(e_pow) - 1)
    ax.set_ylim(6e-15, 3.2)
    fs.clean(ax, grid=True)
    ax.legend(loc="lower left", handletextpad=0.6, borderaxespad=0.5, labelspacing=0.4,
              fontsize=8.4)
    ax.text(0.985, 0.028, "(a)", transform=ax.transAxes, fontsize=9.4, ha="right",
            va="bottom", weight="bold")

    idx = np.arange(1, len(w))
    axb.vlines(idx, 0, np.abs(w[idx]), color=fs.FAINT, lw=1.0, zorder=2)
    axb.plot(idx, np.abs(w[idx]), marker="o", ms=0, ls="none")
    for i in idx:
        c = fs.ACCENT if abs(w[i]) > w[2] - 1e-12 else fs.BLUE
        axb.plot([i], [abs(w[i])], marker="o", ms=5.0, color=c, zorder=4)
    axb.axhline(w[1], color=fs.INK, lw=0.9, ls=(0, (4, 3)), zorder=3)
    fs.label(axb, 10.6, w[1] + 0.035, r"$\mu_1=%.3f$" % w[1], fs=8.4, color=fs.INK,
             ha="right", va="bottom")
    axb.axhline(abs(w[-1]), color=fs.ACCENT, lw=0.8, ls=(0, (1.6, 2.4)), zorder=3)
    fs.label(axb, 10.6, abs(w[-1]) + 0.028, r"$|\mu_{11}|=%.3f$" % abs(w[-1]), fs=8.4,
             color=fs.ACCENT, ha="right", va="bottom")
    fs.label(axb, 2.6, 0.32, r"$|\mu_2|=%.3f$" % abs(w[2]), fs=8.4, color=fs.BLUE,
             ha="left", va="bottom")
    axb.set_xlabel("特征值编号", fontsize=9.0, labelpad=4)
    axb.set_ylabel(r"特征值的绝对值 $|\mu_j|$", fontsize=9.0, labelpad=4)
    axb.set_xlim(0.3, 11.1)
    axb.set_ylim(0, 0.78)
    fs.clean(axb, grid=True)
    axb.text(0.03, 0.985, "(b)", transform=axb.transAxes, fontsize=9.4, ha="left",
             va="top", weight="bold")
    return fs.save(fig, "theory", "ch09", "power-iter")


if __name__ == "__main__":
    fig_sample_error()
    fig_gap_bias()
    fig_sign_flip()
    fig_power_iter()
