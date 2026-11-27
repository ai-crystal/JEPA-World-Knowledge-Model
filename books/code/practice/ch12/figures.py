# -*- coding: utf-8 -*-
"""第 12 章插图：Flickr 三组文本编码器、WIT 规模曲线、42 倍样本效率、误差尺度。

前三张图的数值取自 Balestriero & LeCun, "Proving and Improving the Platonic
Representation Hypothesis with JEPAs"（arXiv:2610.39302）的 Table 1、Table 2、
Table 3 与 Figure 1、Figure 9 及其正文，不做任何插值或补点。
图 12-4 左幅是模拟数据，右幅的五个数取自同一篇论文的 Table 1、Table 3、
第 4.4 节与附录 C。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_flickr_bars():
    """图 9-1：Flickr30k 上图像 JEPA 对四种文本编码器的三条指标（Table 1）。"""
    metrics = ["互 10-NN", "R@1", "R@5"]
    chance = [.010, .001, .005]
    series = [
        ("好视图 JEPA", [.062, .044, .133], fs.ACCENT),
        ("坏视图 JEPA", [.047, .027, .095], fs.BLUE),
        ("下一词预测", [.042, .021, .077], fs.GREEN),
        ("下一词·成对描述", [.043, .023, .083], fs.PINK),
        ("随机初始化", [.010, .001, .011], "#C3CAD3"),
    ]
    n = len(series)
    bw = 0.152
    x0 = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for i, (name, vals, col) in enumerate(series):
        xs = x0 + (i - (n - 1) / 2) * (bw + 0.014)
        ax.bar(xs, vals, width=bw, color=col, label=name, zorder=3,
               edgecolor="white", linewidth=0.6)
        for x, v in zip(xs, vals):
            ax.text(x, max(v + 0.004, 0.007), f"{v:.3f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=7.6, color=fs.MUTED, zorder=4)

    ax.set_xticks(x0)
    ax.set_xticklabels([f"{m}\n随机基线 {c:.3f}".replace("0.", ".") for m, c in zip(metrics, chance)],
                       fontsize=9.4, color=fs.INK)
    ax.set_ylim(0, 0.158)
    ax.set_yticks([0, .04, .08, .12])
    ax.set_ylabel("指标数值（越高越好）", fontsize=9.4)
    fs.clean(ax, grid=True)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=8.8,
              handlelength=1.0, columnspacing=1.1, labelspacing=0.35, borderaxespad=0.2)
    return fs.save(fig, "practice", "ch12", "flickr-bars")


def fig_wit_scaling():
    """图 9-2：WIT-1024 上的互 10-NN 随训练图片数上升（Figure 9 与正文）。"""
    m_good = [2.16, 10, 50, 100]
    v_good = [.108, .111, .125, .125]
    m_bad = [2.16, 50]
    v_bad = [.053, .063]
    m_ntp = [2.16, 50]
    v_ntp = [.071, .079]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.set_xscale("log")
    ax.set_xlim(0.016, 210)
    ax.set_ylim(0.022, 0.145)

    # 语言模型参考线（同一评测协议下的对照，不属于本图的数据点）
    for y, col, lab, dy, va in ((.125, fs.PINK, "Qwen2.5-3B　.125", .0035, "bottom"),
                                (.118, fs.GOLD, "Qwen2.5-7B　.118", -.0035, "top")):
        ax.plot([0.016, 210], [y, y], color=col, lw=1.0, ls=(0, (6, 3)), zorder=1)
        ax.text(0.019, y + dy, lab, ha="left", va=va, fontsize=8.4, color=col)

    ax.plot(m_bad, v_bad, color=fs.BLUE, lw=1.4, ls=(0, (4, 2)), marker="s", ms=4.6,
            zorder=4, markerfacecolor="white", markeredgewidth=1.1, label="坏视图 JEPA")
    ax.plot(m_ntp, v_ntp, color=fs.GREEN, lw=1.4, ls=(0, (2, 2)), marker="^", ms=5.2,
            zorder=4, markerfacecolor="white", markeredgewidth=1.1, label="下一词预测")
    ax.plot(m_good, v_good, color=fs.ACCENT, lw=1.9, marker="o", ms=6.2, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0, label="好视图 JEPA，33M")

    # 逐点标数值：落在参考线上的两个点标在上方，其余标在下方
    for x, v, dy, va in ((2.16, .108, -.0042, "top"), (10, .111, -.0042, "top"),
                         (50, .125, .0045, "bottom"), (100, .125, .0045, "bottom")):
        ax.text(x, v + dy, f"{v:.3f}".lstrip("0"), ha="center", va=va,
                fontsize=8.3, color=fs.MUTED, zorder=6)
    for xs, vs in ((m_bad, v_bad), (m_ntp, v_ntp)):
        for x, v in zip(xs, vs):
            ax.text(x, v - .0042, f"{v:.3f}".lstrip("0"), ha="center", va="top",
                    fontsize=8.3, color=fs.MUTED, zorder=6)

    # Flickr 字幕那一点换过语料，不与网页文本连成一条线
    ax.plot([0.029], [.034], marker="D", ms=5.6, color=fs.ACCENT, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0, label="好视图 JEPA，Flickr 字幕")
    ax.text(0.029, .0375, ".034", ha="center", va="bottom", fontsize=8.3, color=fs.MUTED)

    ax.set_xticks([0.03, 2.16, 10, 50, 100])
    ax.set_xticklabels(["0.03\nFlickr", "2.16", "10", "50", "100"])
    ax.set_xlabel("训练用到的图片数（百万，对数刻度）", fontsize=9.4, labelpad=7)
    ax.set_ylabel("与 DINOv2 ViT-B 的互 10-NN", fontsize=9.4)
    ax.set_yticks([.04, .06, .08, .10, .12, .14])
    fs.clean(ax, grid=True)
    ax.tick_params(axis="x", pad=4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8.6,
              handlelength=1.9, columnspacing=2.4, labelspacing=0.5, borderaxespad=0.0)
    return fs.save(fig, "practice", "ch12", "wit-scaling")


def fig_sample_efficiency():
    """图 9-3：同一个 109M 编码器达到同一对齐水平所需的样本数（4.2 节）。"""
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.set_xscale("log")
    ax.set_xlim(9, 2400)
    ax.set_ylim(0, 0.138)

    pts = [(21.6, .109, fs.ACCENT, "好视图 JEPA"), (900, .111, fs.GREEN, "下一词预测")]

    for m, v, col, name in pts:
        ax.plot([m, m], [0, v], color=col, lw=0.9, ls=(0, (2, 3)), zorder=2)
        ax.plot([m], [v], marker="o", ms=10, color=col, zorder=5,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(m, v + .006, f"{v:.3f}".replace("0.", "."), ha="center", va="bottom",
                fontsize=11.5, color=col, weight="bold")
        ax.text(m, v - .010, name, ha="center", va="top", fontsize=9.2, color=col,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.6), zorder=6)

    ax.plot([21.6, 900], [.109, .111], color=fs.FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.text(139, .1145, "同一对齐水平", ha="center", va="bottom", fontsize=9, color=fs.MUTED)

    ax.annotate("", xy=(21.6, .034), xytext=(900, .034),
                arrowprops=dict(arrowstyle="|-|", color=fs.INK, lw=1.0, mutation_scale=8))
    ax.text(139, .043, "样本少 42 倍", ha="center", va="bottom", fontsize=10.5, color=fs.INK)
    ax.text(139, .027, "词元少 14 倍，唯一图片少 435 倍", ha="center", va="top",
            fontsize=8.8, color=fs.MUTED)

    ax.set_xticks([10, 21.6, 100, 900, 2400])
    ax.set_xticklabels(["10", "21.6", "100", "900", "2400"])
    ax.set_xlabel("训练样本数（百万，对数刻度）", fontsize=9.4, labelpad=7)
    ax.set_ylabel("WIT-1024 互 10-NN", fontsize=9.4)
    ax.set_yticks([0, .04, .08, .12])
    fs.clean(ax)
    ax.tick_params(axis="x", pad=4)
    return fs.save(fig, "practice", "ch12", "sample-efficiency")


# ---------- 图 12-4：误差的尺度 ----------

_SIM_D = 8      # 共享场景变量的维度
_SIM_M = 24     # 两侧嵌入的维度
_SIM_SIG = 4.4  # 噪声强度，调到互 10-NN 落在 0.06 附近
_SIM_K = 10     # 邻居数，与正文一致


def _sim_mnn(phi, psi, k=_SIM_K):
    """在两组嵌入上算互 10-NN，自己不算自己的邻居。"""
    def nn(x):
        g = (x * x).sum(1)
        dd = g[:, None] + g[None, :] - 2 * x @ x.T
        np.fill_diagonal(dd, np.inf)
        return np.argpartition(dd, k, axis=1)[:, :k]

    a, b = nn(phi), nn(psi)
    return np.mean([len(set(a[i]) & set(b[i])) for i in range(len(phi))]) / k


def _fmt(v, nd):
    """把 0.062 写成 .062，与正文一致。"""
    t = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return t[1:] if t.startswith("0.") else t


def fig_error_scale():
    """图 12-4：左幅模拟四种波动来源，右幅排五个实测误差。"""
    rng = np.random.default_rng(12)
    n = 1000
    z = rng.normal(size=(n, _SIM_D))

    def sides():
        """列归一化的随机投影，保证每个种子的信号强度一致。"""
        w_i = rng.normal(0, 1, (_SIM_D, _SIM_M))
        w_t = rng.normal(0, 1, (_SIM_D, _SIM_M))
        return (w_i / np.linalg.norm(w_i, axis=0, keepdims=True),
                w_t / np.linalg.norm(w_t, axis=0, keepdims=True))

    def emb(z, w, scale):
        return z @ w + scale * rng.normal(size=(len(z), _SIM_M)) / np.sqrt(_SIM_M)

    # 一个参照模型：权重与噪声强度固定，测试集固定
    w_i, w_t = sides()
    phi, psi = emb(z, w_i, _SIM_SIG), emb(z, w_t, _SIM_SIG)

    seeds, tests, boot_d, boot_f = [], [], [], []
    for _ in range(80):
        # 换种子：只重抽初始化与噪声，测试集不动
        wi, wt = sides()
        seeds.append(_sim_mnn(emb(z, wi, _SIM_SIG), emb(z, wt, _SIM_SIG)))

        # 重抽测试集：模型不动，换成另一千对场景
        zt = rng.normal(size=(n, _SIM_D))
        tests.append(_sim_mnn(emb(zt, w_i, _SIM_SIG), emb(zt, w_t, _SIM_SIG)))

        # bootstrap 去重：有放回抽编号，只剩约 632 个互不相同的样本
        idx = rng.integers(0, n, n)
        u = np.unique(idx)
        boot_d.append(_sim_mnn(phi[u], psi[u]))

        # bootstrap 补足：把重复点补回 N 个，加极小的扰动打断零距离
        sc = np.sqrt((phi ** 2).sum(1).mean()) * 1e-3
        boot_f.append(_sim_mnn(phi[idx] + rng.normal(0, sc, (n, _SIM_M)),
                               psi[idx] + rng.normal(0, sc, (n, _SIM_M))))

    rows = [
        ("换种子", seeds, fs.ACCENT),
        ("重抽测试集", tests, fs.BLUE),
        ("bootstrap 去重", boot_d, fs.GREEN),
        ("bootstrap 补足 N", boot_f, fs.PINK),
    ]
    ref = float(np.mean(tests))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4),
                             gridspec_kw=dict(width_ratios=[1.12, 1.0], wspace=0.66))
    ax = axes[0]
    for i, (name, vals, col) in enumerate(rows):
        v = np.asarray(vals)
        lo, hi, mu = np.percentile(v, 2.5), np.percentile(v, 97.5), v.mean()
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle="round", zorder=3)
        ax.plot([mu], [y], marker="o", ms=6.2, color=col, markeredgecolor="white",
                markeredgewidth=1.0, zorder=4)
        ax.text(hi + 0.005, y + 0.13, f"{mu:.3f}"[1:], ha="left", va="center",
                fontsize=8.2, color=col)
        sub = ("±" + f"{(hi - lo) / 2:.3f}"[1:]) if i < 2 else ("偏 +" + f"{mu - ref:.3f}"[1:])
        ax.text(hi + 0.005, y - 0.17, sub, ha="left", va="center",
                fontsize=7.4, color=fs.MUTED)

    ax.axvline(ref, color=fs.FAINT, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(ref + 0.002, 3.66, "中性参照", ha="left", va="bottom", fontsize=8.0,
            color=fs.MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.0, color=fs.INK)
    ax.set_ylim(-0.55, 3.88)
    ax.set_xlim(0.045, 0.205)
    ax.set_xlabel("模拟的互 10-NN（95% 区间）", fontsize=9.2)
    fs.clean(ax, grid=False)
    ax.tick_params(axis="y", length=0)

    # 右幅：五个实测误差，对数刻度
    ax = axes[1]
    items = [
        ("种子半极差\nFlickr 互 10-NN", 0.003, fs.ACCENT),
        ("定律预测误差\n85 个未跑的设置", 0.0093, fs.BLUE),
        ("两种子重跑差\n18 次，RMSE", 0.0096, fs.BLUE),
        ("定律在 3.2 亿样本\n处的高估（q=1）", 0.037, fs.GOLD),
        ("第四个种子\n0.184 掉到 0.121", 0.063, fs.PINK),
    ]
    for i, (name, v, col) in enumerate(items):
        y = len(items) - 1 - i
        ax.plot([0.0022, v], [y, y], color=col, lw=1.0, ls=(0, (2, 2)), zorder=2)
        ax.plot([v], [y], marker="o", ms=7.0, color=col, markeredgecolor="white",
                markeredgewidth=1.0, zorder=4)
        ax.text(v * 1.25, y, _fmt(v, 4), ha="left", va="center",
                fontsize=8.2, color=col)

    ax.set_xscale("log")
    ax.set_xlim(0.0022, 0.16)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items][::-1], fontsize=7.6, color=fs.INK)
    ax.set_xticks([0.003, 0.01, 0.03, 0.1])
    ax.set_xticklabels([".003", ".01", ".03", ".1"], fontsize=8.4)
    ax.set_xlabel("误差大小（对数刻度）", fontsize=9.2)
    fs.clean(ax, grid=True)
    ax.tick_params(axis="y", length=0, pad=2)

    se = np.sqrt(_SIM_K * ref) / _SIM_K / np.sqrt(n)   # 泊松近似下的均值标准误
    print("sim ref %.4f seeds %.4f tests %.4f boot_d %.4f boot_f %.4f  poisson-se %.4f"
          % (ref, np.mean(seeds), np.mean(tests), np.mean(boot_d), np.mean(boot_f), se))
    return fs.save(fig, "practice", "ch12", "error-scale")


if __name__ == "__main__":
    fig_flickr_bars()
    fig_wit_scaling()
    fig_sample_efficiency()
    fig_error_scale()
