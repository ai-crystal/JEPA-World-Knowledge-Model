# -*- coding: utf-8 -*-
"""第 9 章插图：时空 tubelet 与掩码、V-JEPA 2 的三段配方与规划回路。"""
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


if __name__ == "__main__":
    fig_tubelet()
    fig_pipeline()
