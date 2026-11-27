# -*- coding: utf-8 -*-
"""第 7 章插图：下一词预测的 JEPA 形式、O(k) 与 GL(k) 的自由度、温度与 margin。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


# ---------------------------------------------------------------- 图 7-1
def fig_next_token_jepa():
    fig = plt.figure(figsize=(6.9, 3.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1.13); ax.set_ylim(0.11, 0.93); ax.axis("off")

    fs.label(ax, 0.020, 0.895, "正样本取自同一序列之内", fs=8.2, color=fs.MUTED, ha="left")

    fs.box(ax, 0.015, 0.565, 0.148, 0.160, "前缀 $c$", fc="#EEF2FF", ec=fs.BLUE, fs=10, lw=1.15)
    fs.label(ax, 0.089, 0.528, "前 $t$ 个词", fs=7.8, color=fs.MUTED)

    fs.box(ax, 0.200, 0.565, 0.150, 0.160, "编码器 $h$", fc="#FFFFFF", ec=fs.INK, fs=9.6, lw=1.15)
    fs.arrow(ax, (0.165, 0.645), (0.198, 0.645), color=fs.INK, lw=1.15)

    fs.box(ax, 0.390, 0.578, 0.112, 0.135, "$h(c)$", fc="#FFFFFF", ec=fs.INK, fs=9.6, lw=1.15)
    fs.label(ax, 0.446, 0.540, "上下文向量", fs=7.6, color=fs.MUTED)
    fs.arrow(ax, (0.352, 0.645), (0.388, 0.645), color=fs.INK, lw=1.15)

    fs.box(ax, 0.598, 0.578, 0.130, 0.135, "$\\langle h, w_v\\rangle$", fc="#F5F3FF",
           ec=fs.PURPLE, fs=10, lw=1.15)
    fs.arrow(ax, (0.504, 0.645), (0.596, 0.645), color=fs.INK, lw=1.15)

    fs.box(ax, 0.390, 0.190, 0.152, 0.135, "词向量表 $\\{w_v\\}$", fc="#FFF7ED",
           ec=fs.ACCENT, fs=9.4, lw=1.15)
    fs.label(ax, 0.466, 0.157, "每个词一行", fs=7.6, color=fs.MUTED)
    fs.arrow(ax, (0.544, 0.262), (0.648, 0.572), color=fs.ACCENT, lw=1.15, rad=-0.14)

    # 预测项
    fs.box(ax, 0.775, 0.645, 0.205, 0.150, "$-\\langle h, w_y\\rangle$\n预测项",
           fc="#FFFFFF", ec=fs.BLUE, fs=9.6, lw=1.15)
    fs.arrow(ax, (0.730, 0.678), (0.773, 0.716), color=fs.BLUE, lw=1.15)

    # 反塌缩项
    fs.box(ax, 0.775, 0.330, 0.205, 0.165,
           "$\\log\\sum_v e^{\\langle h, w_v\\rangle}$\n反塌缩项 $A(h)$",
           fc="#FFFFFF", ec=fs.PURPLE, fs=9.6, lw=1.15)
    fs.arrow(ax, (0.730, 0.612), (0.773, 0.462), color=fs.PURPLE, lw=1.15, rad=0.12)

    # 两项相加得到损失：右侧汇流线 + 求和号
    ax.plot([0.980, 0.997], [0.720, 0.720], color=fs.INK, lw=1.0, zorder=2)
    ax.plot([0.980, 0.997], [0.4125, 0.4125], color=fs.INK, lw=1.0, zorder=2)
    ax.plot([0.997, 0.997], [0.4125, 0.720], color=fs.INK, lw=1.0, zorder=2)
    ax.add_patch(plt.Circle((0.997, 0.566), 0.026, facecolor="white", edgecolor=fs.INK,
                            lw=1.0, zorder=3))
    fs.label(ax, 0.997, 0.566, "$+$", fs=11, color=fs.INK)
    fs.arrow(ax, (1.024, 0.566), (1.070, 0.566), color=fs.INK, lw=1.1, ms=7)
    fs.label(ax, 1.078, 0.566, "损失", fs=9.2, color=fs.INK, ha="left")
    return fs.save(fig, "theory", "ch07", "next-token-jepa")


# ---------------------------------------------------------------- 图 7-2
def _lattice(ax, M, half=0.7, step=0.2, color=fs.GRID, lw=0.6):
    for k in np.arange(-half, half + 1e-9, step):
        for seg in ([[-half, k], [half, k]], [[k, -half], [k, half]]):
            p = np.asarray(seg, float) @ M.T
            ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, zorder=1)
    box = np.asarray([[-half, -half], [-half, half], [half, half],
                      [half, -half], [-half, -half]], float) @ M.T
    ax.plot(box[:, 0], box[:, 1], color=fs.FAINT, lw=0.9, zorder=1)


def _unit_image(ax, M, color=fs.MUTED, ls=(0, (4, 3)), lw=1.0):
    th = np.linspace(0, 2 * np.pi, 240)
    c = np.stack([np.cos(th), np.sin(th)], 1) @ M.T
    ax.plot(c[:, 0], c[:, 1], color=color, ls=ls, lw=lw, zorder=2)


def _triad(ax, M, scale=1.0, pad=0.20):
    A = np.zeros(2)
    B = np.array([1.0, 0.0]) * scale
    C = np.array([0.0, 1.0]) * scale
    A, B, C = A @ M.T, B @ M.T, C @ M.T
    for P, lab in ((B, "$B$"), (C, "$C$")):
        fs.arrow(ax, tuple(A), tuple(P), color=fs.INK, lw=1.5, ms=9)
        u = (P - A) / np.linalg.norm(P - A)
        ax.text(P[0] + pad * u[0], P[1] + pad * u[1], lab, fontsize=9.4,
                color=fs.INK, ha="center", va="center", zorder=6)
    ax.scatter([A[0]], [A[1]], s=26, color=fs.INK, zorder=6)
    ax.text(A[0] - 0.20, A[1] - 0.20, "$A$", fontsize=9.4, color=fs.INK,
            ha="center", va="center", zorder=6)
    return A, B, C


def _stats(A, B, C):
    dAB = np.linalg.norm(B - A)
    dAC = np.linalg.norm(C - A)
    v1, v2 = (B - A) / dAB, (C - A) / dAC
    ang = np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1)))
    return dAB, dAC, dAB / dAC, ang


def fig_gl_vs_o():
    # 左：30 度旋转；右：拉伸加剪切，行列式为正、条件数约 2.4
    th = np.deg2rad(30)
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    G = np.array([[1.35, 0.45], [0.08, 0.70]])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.30))
    titles = [r"(a) 只差一个旋转 $Q\in O(k)$", r"(b) 只差一个可逆映射 $G\in GL(k)$"]
    notes = [("圆还是圆，长度与夹角一个也没变", fs.BLUE),
             ("圆压成椭圆，拉伸与剪切都露了出来", fs.ACCENT)]
    for ax, M, title, (note, col) in zip(axes, (Q, G), titles, notes):
        ax.set_aspect("equal")
        ax.axis("off")
        _lattice(ax, M)
        _unit_image(ax, M)
        A, B, C = _triad(ax, M)
        _, _, ratio, ang = _stats(A, B, C)
        ax.set_xlim(-1.62, 1.62)
        ax.set_ylim(-1.55, 1.62)
        ax.set_title(title, fontsize=10.2, color=fs.INK, pad=4)
        ax.text(0.5, -0.030, f"距离比 $|AB|/|AC|$ = {ratio:.2f}，夹角 = {ang:.0f}" + r"$^\circ$",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.8,
                color=fs.MUTED)
        ax.text(0.5, -0.098, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=9.0, color=col)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.20, wspace=0.06)
    return fs.save(fig, "theory", "ch07", "gl-vs-o")


# ---------------------------------------------------------------- 图 7-3
def fig_temperature_margin():
    s = np.array([2.6, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
    ts = np.logspace(np.log10(0.05), np.log10(3.0), 400)

    def loss(t, sy):
        z = s[None, :] / t[:, None]
        m = z.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
        return t * lse - sy

    fig, ax = plt.subplots(figsize=(6.2, 3.15))
    ax.semilogx(ts, loss(ts, 2.0), color=fs.BLUE, lw=1.9, label="目标词不是最高分")
    ax.semilogx(ts, loss(ts, 2.6), color=fs.GREEN, lw=1.9, label="目标词就是最高分")
    ax.axhline(0.6, color=fs.BLUE, lw=1.0, ls=(0, (4, 3)))
    ax.axhline(0.0, color=fs.GREEN, lw=1.0, ls=(0, (4, 3)))

    ax.text(0.062, 0.675, "margin = .6", fontsize=8.8, color=fs.BLUE,
            ha="left", va="bottom")
    ax.text(0.062, 0.075, "margin = 0", fontsize=8.8, color=fs.GREEN,
            ha="left", va="bottom")
    ax.text(0.062, 0.335, r"低温端两条曲线各自停在自己的 margin 上",
            fontsize=8.6, color=fs.MUTED, ha="left", va="center")

    ax.set_xlabel("温度 $t$")
    ax.set_ylabel(r"损失 $-\langle h, w_y\rangle + t\,A(h/t)$")
    ax.set_xlim(0.05, 3.0)
    ax.set_ylim(-0.10, 1.78)
    fs.clean(ax, grid=True)
    ax.legend(loc="upper left", fontsize=8.6, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.95, borderpad=0.5)
    fig.tight_layout()
    return fs.save(fig, "theory", "ch07", "temperature-margin")


# ---------------------------------------------------------------- 图 7-4
def fig_glk():
    """GL(k) 与 O(k) 的自由度：维数随 k 增长，以及极分解把参数切成两段。"""
    ks = np.arange(2, 1601)
    gl = ks.astype(float) ** 2
    orth = ks * (ks - 1) / 2.0
    spd = ks * (ks + 1) / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.30),
                             gridspec_kw=dict(width_ratios=[1.20, 1.0]))

    ax = axes[0]
    ax.loglog(ks, gl, color=fs.ACCENT, lw=1.9, label=r"$GL(k)$，$k^2$")
    ax.loglog(ks, orth, color=fs.BLUE, lw=1.9, label=r"$O(k)$，$k(k-1)/2$")
    ax.loglog(ks, spd, color=fs.GREEN, lw=1.4, ls=(0, (5, 3)),
              label=r"两者之差，$k(k+1)/2$")

    k0 = 768
    ax.plot([k0, k0], [2.0e5, 1.36e6], color=fs.FAINT, lw=0.9, ls=(0, (2, 3)),
            zorder=1)
    ax.scatter([k0], [k0 ** 2], s=24, color=fs.ACCENT, zorder=6)
    ax.scatter([k0], [k0 * (k0 - 1) / 2], s=24, color=fs.BLUE, zorder=6)
    ax.text(k0, 1.40e6, "$k=768$", fontsize=8.2, color=fs.MUTED, ha="center", va="bottom")
    ax.text(0.975, 0.045, "$k=768$：589,824 对 294,528，\n差 295,296，与旋转本身同量级",
            transform=ax.transAxes, fontsize=8.3, color=fs.INK, ha="right", va="bottom",
            linespacing=1.5)
    ax.set_xlim(2, 2200)
    ax.set_ylim(1, 4e6)
    ax.set_xlabel("嵌入维度 $k$")
    ax.set_ylabel("参数个数")
    ax.set_title("(a) 两个群的维数随 $k$ 增长", fontsize=10.0, color=fs.INK, pad=5)
    ax.set_xticks([2, 10, 100, 1000])
    ax.set_xticklabels(["2", "10", "100", "1000"])
    fs.clean(ax, grid=True)
    ax.legend(loc="upper left", fontsize=8.1, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.92, borderpad=0.42, handlelength=1.5,
              labelspacing=0.42)

    axb = axes[1]
    axb.set_xlim(0, 1.02)
    axb.set_ylim(0, 1)
    axb.axis("off")
    axb.set_title("(b) 极分解把参数切两半", fontsize=10.0, color=fs.INK, pad=5)
    x0, x1 = 0.135, 0.905
    for lab, k, y in (("$k=2$", 2, 0.565), ("$k=768$", 768, 0.265)):
        n_o = k * (k - 1) / 2.0
        n_s = k * (k + 1) / 2.0
        tot = n_o + n_s
        xm = x0 + (x1 - x0) * n_o / tot
        axb.add_patch(plt.Rectangle((x0, y), xm - x0, 0.155, facecolor="#DBEAFE",
                                    edgecolor=fs.BLUE, lw=1.1, zorder=3))
        axb.add_patch(plt.Rectangle((xm, y), x1 - xm, 0.155, facecolor="#FDEBE1",
                                    edgecolor=fs.ACCENT, lw=1.1, zorder=3))
        axb.text(x0 - 0.022, y + 0.078, lab, fontsize=9.4, color=fs.INK,
                 ha="right", va="center")
        axb.text((x0 + xm) / 2, y + 0.096, "旋转", fontsize=8.2, color=fs.BLUE,
                 ha="center", va="center", zorder=4)
        axb.text((x0 + xm) / 2, y + 0.050, f"{n_o:,.0f}", fontsize=9.0, color=fs.BLUE,
                 ha="center", va="center", zorder=4, weight="bold")
        axb.text((xm + x1) / 2, y + 0.096, "拉伸剪切", fontsize=8.2, color=fs.ACCENT,
                 ha="center", va="center", zorder=4)
        axb.text((xm + x1) / 2, y + 0.050, f"{n_s:,.0f}", fontsize=9.0, color=fs.ACCENT,
                 ha="center", va="center", zorder=4, weight="bold")
        axb.text(x0, y - 0.038, f"合计 {tot:,.0f} 个参数", fontsize=8.2, color=fs.MUTED,
                 ha="left", va="top")

    axb.text(0.5, 0.895, "$G=QP$，$Q\\in O(k)$、$P$ 对称正定", fontsize=9.0,
             color=fs.INK, ha="center", va="center")
    axb.text(0.5, 0.075, "两段拼起来才是整个可逆群，\n没有任何一对参数互相重复", fontsize=8.4,
             color=fs.MUTED, ha="center", va="center", linespacing=1.6)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.135, wspace=0.30)
    print("k=2: GL %d, O %d, diff %d" % (4, 1, 3))
    print("k=768: GL %s, O %s, diff %s" % (f"{768 ** 2:,}", f"{768 * 767 // 2:,}",
                                            f"{768 * 769 // 2:,}"))
    return fs.save(fig, "theory", "ch07", "glk")


if __name__ == "__main__":
    fig_next_token_jepa()
    fig_gl_vs_o()
    fig_temperature_margin()
    fig_glk()
