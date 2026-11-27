# -*- coding: utf-8 -*-
"""第 6 章插图：时空 tubelet 与掩码、V-JEPA 2 的三段配方与规划回路。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import to_rgb
import figstyle as fs


def shade(c, f):
    """f>0 变亮，f<0 变暗。"""
    r, g, b = to_rgb(c)
    if f >= 0:
        return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)
    return (r * (1 + f), g * (1 + f), b * (1 + f))


def cube(ax, x0, y0, s, dx, dy, fc, ec, lw=0.7, z=3):
    top = [(x0, y0 + s), (x0 + s, y0 + s), (x0 + s + dx, y0 + s + dy), (x0 + dx, y0 + s + dy)]
    right = [(x0 + s, y0), (x0 + s, y0 + s), (x0 + s + dx, y0 + s + dy), (x0 + s + dx, y0 + dy)]
    front = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]
    ax.add_patch(Polygon(front, closed=True, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon(right, closed=True, facecolor=shade(fc, -0.16), edgecolor=ec,
                         lw=lw, zorder=z))
    ax.add_patch(Polygon(top, closed=True, facecolor=shade(fc, 0.30), edgecolor=ec,
                         lw=lw, zorder=z))
    return


def fig_tubelet():
    fig = plt.figure(figsize=(6.9, 3.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    nx, ny, nt = 4, 3, 4
    W, D = 0.380, 0.240
    kx, ky = 0.60 * D, 0.40 * D
    dx, dy = 0.058, 0.143
    th, mu, mv = 0.034, 0.042, 0.048
    ox, oy = 0.098, 0.158

    big = {(0, 2), (1, 2)}
    small = {(3, 0)}

    def pt(k, u, v):
        return (ox + k * dx + u * W + v * kx, oy + k * dy + v * ky)

    for k in range(nt):
        for j in range(ny - 1, -1, -1):
            for i in range(nx):
                u0, u1 = i / nx + mu, (i + 1) / nx - mu
                v0, v1 = j / ny + mv, (j + 1) / ny - mv
                if (i, j) in big:
                    fc, ec = fs.ACCENT, "#7A2707"
                elif (i, j) in small:
                    fc, ec = fs.BLUE, "#132C6E"
                else:
                    fc, ec = "#EDF1F6", "#8E959F"
                c00, c10 = pt(k, u0, v0), pt(k, u1, v0)
                c11, c01 = pt(k, u1, v1), pt(k, u0, v1)
                z0 = 3 + k * 30 + (ny - j) * 4
                # 前侧面
                ax.add_patch(Polygon([c00, c10, (c10[0], c10[1] - th), (c00[0], c00[1] - th)],
                                     closed=True, facecolor=shade(fc, -0.20), edgecolor=ec,
                                     lw=0.6, zorder=z0))
                # 右侧面
                ax.add_patch(Polygon([c10, c11, (c11[0], c11[1] - th), (c10[0], c10[1] - th)],
                                     closed=True, facecolor=shade(fc, -0.34), edgecolor=ec,
                                     lw=0.6, zorder=z0 + 1))
                # 顶面
                ax.add_patch(Polygon([c00, c10, c11, c01], closed=True,
                                     facecolor=shade(fc, 0.10), edgecolor=ec,
                                     lw=0.6, zorder=z0 + 2))

    # 时间轴
    ax.annotate("", xy=(0.052, 0.640), xytext=(0.052, 0.140),
                arrowprops=dict(arrowstyle="-|>", color=fs.MUTED, lw=1.2,
                                mutation_scale=9))
    fs.label(ax, 0.052, 0.676, "时间", fs=8.8, color=fs.MUTED)
    for k in range(nt):
        yy = oy + k * dy + 0.008
        ax.plot([0.048, 0.060], [yy, yy], color=fs.FAINT, lw=0.8, zorder=1)
    fs.label(ax, 0.030, 0.158, "$t_1$", fs=8.0, color=fs.MUTED, ha="right")
    fs.label(ax, 0.030, 0.158 + 3 * dy, "$t_4$", fs=8.0, color=fs.MUTED, ha="right")

    # 大块掩码
    fs.label(ax, 0.150, 0.838, "大块掩码，沿时间轴盖住整列", fs=8.6, color=fs.ACCENT,
             ha="left")
    ax.annotate("", xy=(0.400, 0.655), xytext=(0.255, 0.820),
                arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=0.9,
                                mutation_scale=8, connectionstyle="arc3,rad=-0.20"))
    # 小块掩码
    ax.annotate("", xy=(0.664, 0.584), xytext=(0.714, 0.474),
                arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=0.9,
                                mutation_scale=8))
    fs.label(ax, 0.720, 0.468, "小块掩码", fs=8.6, color=fs.BLUE, ha="left")

    # 说明
    fs.label(ax, 0.098, 0.072, "每层是 2 帧，四层共 8 帧。一个格子是一个 tubelet，对应 16 × 16 像素。",
             fs=8.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.098, 0.024, "被盖住的格子不进入编码器，它们在时间上的对应位置交给预测器",
             fs=8.4, color=fs.MUTED, ha="left")
    return fs.save(fig, "basics", "ch06", "tubelet-mask")


def fig_pipeline():
    fig = plt.figure(figsize=(6.9, 3.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0.062, 1); ax.axis("off")

    # ---------- 上半：三段训练 ----------
    fs.label(ax, 0.045, 0.966, "训练", fs=8.4, color=fs.MUTED, ha="left",
             weight="bold")
    w, h, ytop, gap = 0.255, 0.150, 0.790, 0.072
    xs = [0.045 + i * (w + gap) for i in range(3)]

    stage = [
        ("一　无动作视频预训练", "100 万小时以上视频", "ViT-g，10 亿参数", fs.BLUE),
        ("二　与语言模型对齐", "Llama 3.1 8B", "视频问答", fs.GREEN),
        ("三　动作条件后训练", "Droid 不到 62 小时", "学一个新的预测器", fs.PURPLE),
    ]
    for x, (t1, t2, t3, c) in zip(xs, stage):
        fs.box(ax, x, ytop, w, h, "", fc="#FFFFFF", ec=c, lw=1.2)
        fs.label(ax, x + 0.012, ytop + h - 0.034, t1, fs=8.8, color=c,
                 ha="left", weight="bold")
        fs.label(ax, x + 0.012, ytop + h - 0.078, t2, fs=8.2, color=fs.INK, ha="left")
        fs.label(ax, x + 0.012, ytop + h - 0.118, t3, fs=8.2, color=fs.MUTED, ha="left")

    for i in range(2):
        fs.arrow(ax, (xs[i] + w + 0.006, ytop + h / 2), (xs[i + 1] - 0.006, ytop + h / 2),
                 color=fs.FAINT, lw=1.1)

    # 分支：冻结编码器 + 探测
    x1c = xs[0] + w / 2
    ax.annotate("", xy=(x1c, 0.638), xytext=(x1c, ytop - 0.006),
                arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.0,
                                mutation_scale=8, linestyle=(0, (3, 2))))
    fs.box(ax, xs[0] + 0.006, 0.502, w - 0.012, 0.136, "", fc="#FFF7F0",
           ec=fs.ACCENT, lw=1.0)
    fs.label(ax, x1c, 0.612, "冻结编码器，只训练探测头", fs=8.0,
             color=fs.ACCENT, weight="bold")
    fs.label(ax, x1c, 0.576, "运动分类 77.3", fs=7.8, color=fs.MUTED)
    fs.label(ax, x1c, 0.544, "动作预期 39.7", fs=7.8, color=fs.MUTED)

    # 分支：语言对齐的产出
    x2c = xs[1] + w / 2
    ax.annotate("", xy=(x2c, 0.650), xytext=(x2c, ytop - 0.006),
                arrowprops=dict(arrowstyle="-|>", color=fs.GREEN, lw=1.0,
                                mutation_scale=8, linestyle=(0, (3, 2))))
    fs.label(ax, x2c, 0.626, "PerceptionTest 84.0", fs=8.0, color=fs.GREEN)
    fs.label(ax, x2c, 0.586, "TempCompass 76.9", fs=8.0, color=fs.GREEN)

    # ---------- 下半：规划回路 ----------
    ax.add_patch(plt.Rectangle((0.030, 0.098), 0.940, 0.382, facecolor="#FAFBFC",
                               edgecolor=fs.GRID, lw=1.0, zorder=0))
    fs.label(ax, 0.050, 0.442, "部署　潜空间里的规划回路", fs=8.4, color=fs.MUTED,
             ha="left", weight="bold")
    x3c = xs[2] + w / 2
    ax.annotate("", xy=(x3c, 0.486), xytext=(x3c, ytop - 0.006),
                arrowprops=dict(arrowstyle="-|>", color=fs.PURPLE, lw=1.0,
                                mutation_scale=8, linestyle=(0, (3, 2))))

    w2, h2, yloop, gap2 = 0.200, 0.140, 0.250, 0.033
    xs2 = [0.045 + i * (w2 + gap2) for i in range(4)]
    loop = [
        ("当前帧与目标帧", "编码 $z_k$，$z_g$", fs.INK),
        ("采样 800 条", "动作序列", fs.BLUE),
        ("预测 $T$ 步后的表征", "与 $z_g$ 比 $L_1$", fs.ACCENT),
        ("取能量最低的一条", "只执行第一步", fs.GREEN),
    ]
    for x, (t1, t2, c) in zip(xs2, loop):
        fs.box(ax, x, yloop, w2, h2, "", fc="#FFFFFF", ec=c, lw=1.1, z=3)
        fs.label(ax, x + w2 / 2, yloop + h2 - 0.048, t1, fs=8.2, color=fs.INK)
        fs.label(ax, x + w2 / 2, yloop + h2 - 0.096, t2, fs=8.0, color=c)
    for i in range(3):
        fs.arrow(ax, (xs2[i] + w2 + 0.004, yloop + h2 / 2),
                 (xs2[i + 1] - 0.004, yloop + h2 / 2), color=fs.FAINT, lw=1.1)

    # 回环
    xl, xr = xs2[0] + w2 / 2, xs2[3] + w2 / 2
    ax.annotate("", xy=(xl, yloop - 0.006), xytext=(xl, 0.202),
                arrowprops=dict(arrowstyle="-|>", color=fs.MUTED, lw=1.0,
                                mutation_scale=8, linestyle=(0, (3, 2))))
    ax.plot([xl, xr], [0.202, 0.202], color=fs.MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.plot([xr, xr], [0.202, yloop - 0.006], color=fs.MUTED, lw=1.0, ls=(0, (3, 2)))
    fs.label(ax, (xl + xr) / 2, 0.172, "每一步都重新观测，重新规划", fs=8.0,
             color=fs.MUTED)
    return fs.save(fig, "basics", "ch06", "vjepa2-pipeline")


# ---------------------------------------------------------------- 图 6-3
def fig_fusion():
    fig = plt.figure(figsize=(7.0, 3.25))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0.22, 1); ax.axis("off")

    cw, ch, gx = 0.024, 0.040, 0.005
    fcol = [fs.BLUE, fs.ACCENT, fs.GREEN]
    light = ["#DCE6FB", "#FBE0D2", "#D6EDE4"]

    # ---------------- (a) 早融合
    x0 = 0.040
    fs.label(ax, 0.168, 0.918, "(a) 早融合", fs=9.4, color=fs.INK, weight="bold")
    for i in range(9):
        k = i // 3
        ax.add_patch(Rectangle((x0 + i * (cw + gx), 0.620), cw, ch,
                               facecolor=light[k], edgecolor=fcol[k], lw=0.7, zorder=3))
    for k in range(3):
        xa = x0 + 3 * k * (cw + gx)
        xb = xa + 3 * cw + 2 * gx
        fs.bracket(ax, xa, xb, 0.572, "", color=fcol[k])
        fs.label(ax, (xa + xb) / 2, 0.538, "帧 %d" % (k + 1), fs=7.4, color=fcol[k])
    xa, xb = x0, x0 + 9 * cw + 8 * gx
    fs.bracket(ax, xa, xb, 0.706, "", color=fs.MUTED)
    fs.label(ax, (xa + xb) / 2, 0.734, "一条序列，每两个词元都算相关", fs=7.6,
             color=fs.MUTED)
    fs.label(ax, 0.168, 0.402, "帧与帧的边界只剩位置编码这一条线索", fs=7.6,
             color=fs.MUTED)
    fs.label(ax, 0.168, 0.352, "注意力之后，时间与空间已经混在一起", fs=7.6,
             color=fs.MUTED)

    # ---------------- (b) 晚融合
    fs.label(ax, 0.500, 0.918, "(b) 晚融合", fs=9.4, color=fs.INK, weight="bold")
    xb0 = 0.424
    for k, yy in enumerate([0.690, 0.590, 0.490]):
        for i in range(3):
            ax.add_patch(Rectangle((xb0 + i * (cw + gx), yy), cw, ch,
                                   facecolor=light[k], edgecolor=fcol[k], lw=0.7, zorder=3))
        fs.arrow(ax, (xb0 + 3 * cw + 2 * gx + 0.007, yy + ch / 2),
                 (0.538, yy + ch / 2), color=fcol[k], lw=0.8, ms=7)
    fs.box(ax, 0.542, 0.472, 0.084, 0.256, "", fc="#FFFFFF", ec=fs.MUTED, lw=0.9)
    fs.label(ax, 0.584, 0.600, "各自编码\n各自池化", fs=7.6, color=fs.MUTED)
    fs.arrow(ax, (0.584, 0.466), (0.584, 0.414), color=fs.MUTED, lw=0.9, ms=7)
    fs.box(ax, 0.528, 0.344, 0.112, 0.070, "合并，再预测", fc="#F7F8FA",
           ec=fs.MUTED, lw=0.9, fs=7.6)
    fs.label(ax, 0.500, 0.284, "帧内的冗余没法被时间注意力用上", fs=7.6, color=fs.MUTED)
    fs.label(ax, 0.500, 0.236, "好处是每一帧可以独立并行", fs=7.6, color=fs.MUTED)

    # ---------------- (c) 分解注意力
    fs.label(ax, 0.835, 0.918, "(c) 分解注意力", fs=9.4, color=fs.INK, weight="bold")
    cw2, ch2, gg, vg = 0.032, 0.042, 0.032, 0.056
    gx0, gy0 = 0.742, 0.766
    for r in range(3):
        for c in range(3):
            ax.add_patch(Rectangle((gx0 + c * (cw2 + gg), gy0 - r * (ch2 + vg)), cw2,
                                   ch2, facecolor=light[r], edgecolor=fcol[r], lw=0.7,
                                   zorder=3))
    # 行内箭头只画在格子之间的缝里
    for r in range(3):
        yy = gy0 - r * (ch2 + vg) + ch2 / 2
        for c in range(2):
            xl = gx0 + c * (cw2 + gg) + cw2 + 0.003
            ax.annotate("", xy=(xl + gg - 0.006, yy), xytext=(xl, yy),
                        arrowprops=dict(arrowstyle="<|-|>", color=fs.BLUE, lw=0.7,
                                        mutation_scale=6))
    for c in range(3):
        xx = gx0 + c * (cw2 + gg) + cw2 / 2
        for r in range(2):
            yt = gy0 - r * (ch2 + vg) - 0.003
            ax.annotate("", xy=(xx, yt - vg + 0.006), xytext=(xx, yt),
                        arrowprops=dict(arrowstyle="<|-|>", color=fs.ACCENT, lw=0.7,
                                        mutation_scale=6))
    fs.label(ax, 0.818, 0.430, "蓝色走行，橙色走列", fs=7.6, color=fs.MUTED)
    fs.label(ax, 0.818, 0.380, "一遍在帧内，一遍在同一位置跨帧", fs=7.6, color=fs.MUTED)
    fs.label(ax, 0.818, 0.330, "开销随帧数线性增长", fs=7.6, color=fs.MUTED)
    return fs.save(fig, "basics", "ch06", "fusion")


# ---------------------------------------------------------------- 图 6-4
def fig_action():
    fig = plt.figure(figsize=(7.0, 3.10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0.14, 1); ax.axis("off")

    # ---------- 左：执行器与位姿轴
    fs.label(ax, 0.130, 0.936, "末端的位姿 $\\mathbf{s}_k$", fs=8.8, color=fs.INK,
             weight="bold")
    for xf, xw in ((0.062, 0.014), (0.104, 0.014)):
        ax.add_patch(Rectangle((xf, 0.560), xw, 0.185, facecolor="#E8EBEF",
                               edgecolor=fs.INK, lw=0.9, zorder=4))
    ax.add_patch(Rectangle((0.056, 0.740), 0.070, 0.036, facecolor="#D7DCE3",
                           edgecolor=fs.INK, lw=0.9, zorder=4))
    ax.add_patch(Rectangle((0.077, 0.776), 0.028, 0.062, facecolor="#C9CFD8",
                           edgecolor=fs.INK, lw=0.9, zorder=4))
    ax.annotate("", xy=(0.052, 0.648), xytext=(0.008, 0.648),
                arrowprops=dict(arrowstyle="<|-|>", color=fs.GREEN, lw=0.9,
                                mutation_scale=7))
    fs.label(ax, 0.032, 0.618, "开合", fs=7.4, color=fs.GREEN)

    # ---------- 坐标轴与姿态
    p = (0.205, 0.520)
    ax.annotate("", xy=(p[0] + 0.112, p[1] + 0.014), xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=1.1, mutation_scale=8))
    fs.label(ax, p[0] + 0.124, p[1] + 0.012, "$\\Delta x$", fs=8.2, color=fs.BLUE, ha="left")
    ax.annotate("", xy=(p[0], p[1] + 0.108), xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=1.1, mutation_scale=8))
    fs.label(ax, p[0], p[1] + 0.130, "$\\Delta y$", fs=8.2, color=fs.BLUE)
    ax.annotate("", xy=(p[0] - 0.074, p[1] - 0.072), xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=1.1, mutation_scale=8))
    fs.label(ax, p[0] - 0.082, p[1] - 0.086, "$\\Delta z$", fs=8.2, color=fs.BLUE, ha="right")
    ax.scatter([p[0]], [p[1]], s=18, color=fs.INK, zorder=6)
    from matplotlib.patches import Arc
    ax.add_patch(Arc(p, 0.170, 0.136, theta1=6, theta2=96, color=fs.ACCENT, lw=1.0,
                     zorder=5))
    ax.annotate("", xy=(p[0] + 0.020, p[1] + 0.068), xytext=(p[0] + 0.032, p[1] + 0.053),
                arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.0, mutation_scale=7))
    fs.label(ax, 0.246, 0.610, "姿态 3 维", fs=7.6, color=fs.ACCENT, ha="left")
    fs.label(ax, 0.130, 0.330, "位置以基座为原点，单位是米", fs=7.6, color=fs.MUTED)
    fs.label(ax, 0.130, 0.276, "姿态是绕三个轴的外旋欧拉角", fs=7.6, color=fs.MUTED)

    # ---------- 右：7 维动作向量
    fs.label(ax, 0.700, 0.936, "一个动作 $\\mathbf{a}_k$", fs=8.8, color=fs.INK,
             weight="bold")
    labs = ["$\\Delta x$", "$\\Delta y$", "$\\Delta z$",
            "$\\Delta\\phi$", "$\\Delta\\theta$", "$\\Delta\\psi$", "$g$"]
    cols = [fs.BLUE] * 3 + [fs.ACCENT] * 3 + [fs.GREEN]
    fcs = ["#DCE6FB"] * 3 + ["#FBE0D2"] * 3 + ["#D6EDE4"]
    w, h, gg2, x0c = 0.058, 0.086, 0.007, 0.500
    for i in range(7):
        x = x0c + i * (w + gg2)
        ax.add_patch(Rectangle((x, 0.640), w, h, facecolor=fcs[i], edgecolor=cols[i],
                               lw=1.0, zorder=3))
        fs.label(ax, x + w / 2, 0.683, labs[i], fs=8.4, color=cols[i])
    groups = [((x0c, x0c + 3 * w + 2 * gg2), "位置增量 3 维", fs.BLUE),
              ((x0c + 3 * (w + gg2), x0c + 6 * w + 5 * gg2), "姿态增量 3 维", fs.ACCENT),
              ((x0c + 6 * (w + gg2), x0c + 7 * w + 6 * gg2), "夹爪 1 维", fs.GREEN)]
    for (a, b), txt, c in groups:
        fs.bracket(ax, a, b, 0.756, txt, color=c, fs=7.4, dy=0.030)
    fs.label(ax, x0c + 1.5 * (w + gg2), 0.598, "米", fs=7.2, color=fs.MUTED)
    fs.label(ax, x0c + 4.5 * (w + gg2), 0.598, "弧度", fs=7.2, color=fs.MUTED)
    fs.label(ax, x0c + 6.5 * (w + gg2), 0.598, "0 到 1", fs=7.2, color=fs.MUTED)

    fs.label(ax, 0.700, 0.494, "$\\mathbf{a}_k = \\mathbf{s}_{k+1} - \\mathbf{s}_k$", fs=9.6)
    fs.label(ax, 0.700, 0.410, "同一个增量，从不同起点出发会落到不同位置", fs=7.6,
             color=fs.MUTED)
    fs.label(ax, 0.700, 0.358, "所以动作必须和当前状态一起送进预测器", fs=7.6, color=fs.MUTED)
    fs.label(ax, 0.700, 0.268, "换成关节角，维度与量程随机械臂而变", fs=7.6, color=fs.MUTED)
    fs.label(ax, 0.700, 0.216, "末端位姿写在任务空间里，跨本体通用", fs=7.6, color=fs.MUTED)
    return fs.save(fig, "basics", "ch06", "action-vector")


if __name__ == "__main__":
    fig_tubelet()
    fig_pipeline()
    fig_fusion()
    fig_action()
