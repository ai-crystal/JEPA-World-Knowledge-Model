# -*- coding: utf-8 -*-
"""第 6 章插图：时空 tubelet 与掩码、V-JEPA 2 的三段配方与规划回路、
视频数据管线、评估任务矩阵。"""
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
    return fs.save(fig, "practice", "ch06", "tubelet-mask")


def fig_pipeline():
    fig = plt.figure(figsize=(6.9, 3.72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0.070, 1.005); ax.axis("off")

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
    ax.annotate("", xy=(x1c, 0.650), xytext=(x1c, ytop - 0.006),
                arrowprops=dict(arrowstyle="-|>", color=fs.ACCENT, lw=1.0,
                                mutation_scale=8, linestyle=(0, (3, 2))))
    fs.box(ax, xs[0] + 0.006, 0.548, w - 0.012, 0.100, "", fc="#FFF7F0",
           ec=fs.ACCENT, lw=1.0)
    fs.label(ax, x1c, 0.618, "冻结编码器，只训练探测头", fs=8.0,
             color=fs.ACCENT, weight="bold")
    fs.label(ax, x1c, 0.578, "衡量表征质量，不改变表征", fs=7.8, color=fs.MUTED)

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
    return fs.save(fig, "practice", "ch06", "vjepa2-pipeline")


def fig_double_mask():
    fig = plt.figure(figsize=(6.9, 3.52))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    n, cw, chh = 9, 0.028, 0.0550

    def grid(x0, y0, blocks, color):
        cnt = {}
        for (bi, bj, bw, bh) in blocks:
            for i in range(bi, bi + bw):
                for j in range(bj, bj + bh):
                    if 0 <= i < n and 0 <= j < n:
                        cnt[(i, j)] = cnt.get((i, j), 0) + 1
        for j in range(n):
            for i in range(n):
                k = cnt.get((i, j), 0)
                if k == 0:
                    fc = "#EDF1F6"
                elif k == 1:
                    fc = shade(color, 0.55)
                else:
                    fc = color
                ax.add_patch(plt.Rectangle((x0 + i * cw, y0 + j * chh), cw, chh,
                                           facecolor=fc, edgecolor="#FFFFFF", lw=0.5,
                                           zorder=2))
        ax.add_patch(plt.Rectangle((x0, y0), n * cw, n * chh, facecolor="none",
                                   edgecolor=fs.FAINT, lw=0.9, zorder=7))
        return len(cnt) / (n * n)

    short = [(0, 0, 3, 4), (2, 1, 4, 3), (5, 0, 3, 4), (1, 3, 4, 3),
             (4, 3, 4, 3), (6, 4, 3, 4), (0, 5, 4, 3), (3, 6, 4, 3)]
    long_ = [(0, 1, 8, 7), (1, 2, 8, 7)]

    r1 = grid(0.075, 0.300, short, fs.ACCENT)
    r2 = grid(0.375, 0.300, long_, fs.BLUE)
    print(f"短程掩码覆盖 {r1:.0%}，长程掩码覆盖 {r2:.0%}（块数与面积按论文比例摆放）")

    fs.label(ax, 0.075, 0.885, "(a)　短程掩码　8 块", fs=9.0, color=fs.ACCENT, ha="left")
    fs.label(ax, 0.075, 0.833, "每块约占单帧面积的 15%", fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax, 0.375, 0.885, "(b)　长程掩码　2 块", fs=9.0, color=fs.BLUE, ha="left")
    fs.label(ax, 0.375, 0.833, "每块约占单帧面积的 70%", fs=8.2, color=fs.MUTED, ha="left")

    # (c) 一次迭代的前向账
    fs.label(ax, 0.680, 0.885, "(c)　一次迭代的前向账", fs=9.0, color=fs.INK, ha="left")
    rows = [("上下文编码器", "2 次", fs.BLUE), ("预测器", "2 次", fs.BLUE),
            ("目标编码器", "1 次", fs.GREEN)]
    for k, (name, times, col) in enumerate(rows):
        yy = 0.730 - k * 0.108
        fs.box(ax, 0.668, yy, 0.300, 0.086, "", fc="#FFFFFF", ec=fs.GRID, lw=1.0, z=3)
        fs.label(ax, 0.684, yy + 0.043, name, fs=8.2, color=fs.INK, ha="left")
        fs.label(ax, 0.952, yy + 0.043, times, fs=8.6, color=col, ha="right", weight="bold")
    fs.label(ax, 0.680, 0.378, "两份掩码各有一份上下文编码器与预测器", fs=8.0,
             color=fs.MUTED, ha="left")
    fs.label(ax, 0.680, 0.332, "目标表征只算一遍，两份损失共用", fs=8.0,
             color=fs.GREEN, ha="left")

    fs.label(ax, 0.075, 0.208, "一格是一个 tubelet 在某一帧上的投影，浅色格被盖住一次，深色格两次以上",
             fs=8.2, color=fs.MUTED, ha="left")
    fs.label(ax, 0.075, 0.152, "两份掩码平均遮掉约 90% 的格子，差别只在块的个数与大小",
             fs=8.2, color=fs.ACCENT, ha="left")
    fs.label(ax, 0.075, 0.096, "格子只用来数覆盖，真实掩码在 14 乘 14 的词元网格上采样",
             fs=8.2, color=fs.MUTED, ha="left")
    return fs.save(fig, "practice", "ch06", "double-mask")


def fig_video_data():
    """视频数据管线的四道关，以及 VM22M 的样本占比与采样权重。"""
    fig = plt.figure(figsize=(6.9, 4.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------- 上半：四道关 ----------
    fs.label(ax, 0.048, 0.966, "(a)　一段视频到一批张量", fs=8.8, color=fs.MUTED,
             ha="left", weight="bold")
    w, h, y0, gap = 0.207, 0.206, 0.728, 0.024
    xs = [0.050 + i * (w + gap) for i in range(4)]
    stage = [
        ("一　采集与去重", "剔掉验证集重叠", "2200 万段入库", fs.BLUE),
        ("二　解码与抽帧", "4 fps 取 64 帧", "一段 16 秒", fs.PURPLE),
        ("三　裁剪与缩放", "缩放 0.3 至 1.0", "裁到 256 × 256", fs.GREEN),
        ("四　混样与成批", "按来源加权抽", "3072 段一批", fs.ACCENT),
    ]
    for x, (t1, t2, t3, c) in zip(xs, stage):
        fs.box(ax, x, y0, w, h, "", fc="#FFFFFF", ec=c, lw=1.1)
        fs.label(ax, x + 0.013, y0 + h - 0.040, t1, fs=8.2, color=c, ha="left",
                 weight="bold")
        fs.label(ax, x + 0.013, y0 + h - 0.098, t2, fs=7.2, color=fs.INK, ha="left")
        fs.label(ax, x + 0.013, y0 + h - 0.146, t3, fs=7.2, color=fs.MUTED, ha="left")

    for i in range(3):
        fs.arrow(ax, (xs[i] + w + 0.003, y0 + h / 2), (xs[i + 1] - 0.003, y0 + h / 2),
                 color=fs.FAINT, lw=1.1)

    fs.label(ax, 0.050, 0.676, "顺序固定：去重必须在裁剪之前，抽帧率决定后面每一步的量",
             fs=7.4, color=fs.MUTED, ha="left")

    # ---------- 下半：配比 ----------
    fs.label(ax, 0.048, 0.610, "(b)　VM22M 五个来源的样本占比与采样权重", fs=8.8,
             color=fs.MUTED, ha="left", weight="bold")
    ax.add_patch(plt.Rectangle((0.700, 0.598), 0.030, 0.014, facecolor="#DCE2EA",
                               edgecolor="none", zorder=3))
    fs.label(ax, 0.736, 0.605, "样本占比", fs=7.0, color=fs.MUTED, ha="left")
    ax.add_patch(plt.Rectangle((0.812, 0.598), 0.030, 0.014, facecolor=fs.ACCENT,
                               edgecolor="none", zorder=3))
    fs.label(ax, 0.848, 0.605, "权重", fs=7.0, color=fs.ACCENT, ha="left")

    src = [("SSv2", 16.8, 0.056), ("Kinetics", 73.3, 0.188),
           ("HowTo100M", 110.0, 0.318), ("YT1B", 1900.0, 0.188),
           ("ImageNet", 100.0, 0.250)]
    total, x0, factor, bh = 2200.0, 0.250, 0.550, 0.028
    ys = [0.546, 0.458, 0.370, 0.282, 0.194]
    for (name, n, wgt), y in zip(src, ys):
        share = n / total
        ax.add_patch(plt.Rectangle((x0, y), share * factor, bh, facecolor="#DCE2EA",
                                   edgecolor="none", zorder=2))
        ax.add_patch(plt.Rectangle((x0, y - bh - 0.010), wgt * factor, bh,
                                   facecolor=fs.ACCENT, edgecolor="none", zorder=2))
        fs.label(ax, x0 - 0.012, y - 0.014, name, fs=7.8, color=fs.INK, ha="right")
        fs.label(ax, x0 - 0.012, y - 0.046, f"{n:g} 万段", fs=6.6, color=fs.MUTED,
                 ha="right")
        fs.label(ax, x0 + wgt * factor + 0.008, y - bh - 0.010 + bh / 2,
                 f"{wgt:.0%}", fs=6.8, color=fs.ACCENT, ha="left")

    fs.label(ax, 0.868, 0.560, "每轮遍数", fs=7.2, color=fs.MUTED, ha="center")
    for (name, n, wgt), y in zip(src, ys):
        ep = wgt * total / n
        col = fs.ACCENT if ep > 1 else fs.MUTED
        fs.label(ax, 0.868, y - 0.020, f"{ep:.1f} 遍", fs=8.0, color=col, ha="center",
                 weight="bold" if ep > 5 else "normal")

    # 坐标轴
    ax.plot([x0, x0 + 0.80 * factor], [0.148, 0.148], color=fs.FAINT, lw=0.9)
    for t in (0, 0.25, 0.5, 0.75):
        xx = x0 + t * factor
        ax.plot([xx, xx], [0.148, 0.142], color=fs.FAINT, lw=0.8)
        fs.label(ax, xx, 0.120, f"{t:.0%}" if t else "0", fs=6.8, color=fs.MUTED)
        ax.plot([xx, xx], [0.152, 0.560], color=fs.GRID, lw=0.6, zorder=0)

    fs.label(ax, 0.048, 0.070, "权重之和为 1；权重乘 2200 万再除以来源样本数，就是一轮里被走过的遍数。",
             fs=7.4, color=fs.INK, ha="left")
    fs.label(ax, 0.048, 0.030, "YT1B 一轮只走两成，几个小数据集被重复看五到七遍。",
             fs=7.4, color=fs.ACCENT, ha="left")

    for (_, n, wgt), y in zip(src, ys):
        print(f"{n:7.1f} 万段 占 {n / total:6.1%}，权重 {wgt:.3f}，一轮走 {wgt * total / n:.2f} 遍")
    return fs.save(fig, "practice", "ch06", "video-pipeline")


def fig_eval_matrix():
    """四个基准的评估矩阵，以及两代最终模型在六个分类任务上的差。"""
    fig = plt.figure(figsize=(6.9, 3.72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fs.label(ax, 0.030, 0.952, "(a)　四个基准各在考什么", fs=8.8, color=fs.MUTED,
             ha="left", weight="bold")

    cx = [0.038, 0.120, 0.205, 0.372, 0.548]
    heads = ["基准", "类别", "考什么", "评估输入", "V-JEPA 2"]
    has = ["left", "left", "left", "left", "right"]
    for x, t, ha in zip(cx, heads, has):
        fs.label(ax, x, 0.884, t, fs=7.4, color=fs.MUTED, ha=ha)
    ax.plot([0.030, 0.556], [0.866, 0.866], color=fs.FAINT, lw=0.9)

    rows = [
        ("SSv2", "运动", "动作本身，物体可换", "16 × 2 × 3", "77.3", fs.BLUE),
        ("K400", "外观", "动作类别，单帧可猜", "16 × 8 × 3", "87.3", fs.GREEN),
        ("IN1K", "外观", "物体，时间维无用", "16 × 5 × 3", "85.1", fs.GREEN),
        ("EK100", "预期", "下一秒会发生什么", "32 帧 8 fps", "39.7", fs.PURPLE),
    ]
    ys = [0.782, 0.662, 0.542, 0.422]
    for (name, kind, what, inp, score, c), y in zip(rows, ys):
        ax.add_patch(plt.Rectangle((0.024, y - 0.030), 0.005, 0.058, facecolor=c,
                                   edgecolor="none", zorder=3))
        fs.label(ax, cx[0], y, name, fs=8.2, color=fs.INK, ha="left", weight="bold")
        fs.label(ax, cx[1], y, kind, fs=7.4, color=c, ha="left")
        fs.label(ax, cx[2], y, what, fs=7.2, color=fs.INK, ha="left")
        fs.label(ax, cx[3], y, inp, fs=7.2, color=fs.MUTED, ha="left")
        fs.label(ax, cx[4], y, score, fs=7.8, color=c, ha="right", weight="bold")
    ax.plot([0.030, 0.556], [0.378, 0.378], color=fs.FAINT, lw=0.9)

    fs.label(ax, 0.030, 0.338, "分数取 V-JEPA 2 ViT-g384，除 EK100 用召回前五之外都是 top-1。",
             fs=6.9, color=fs.MUTED, ha="left")
    fs.label(ax, 0.030, 0.300, "同一个 ViT-H 在 SSv2 上按 16 × 1 × 1 读 72.2，按 16 × 2 × 3 读 74.3。",
             fs=6.9, color=fs.MUTED, ha="left")

    # 冻结编码器加探测头的协议说明
    fs.box(ax, 0.026, 0.086, 0.530, 0.170, "", fc="#FAFBFC", ec=fs.GRID, lw=0.9, z=1)
    fs.label(ax, 0.040, 0.224, "探测头协议（六个分类任务共用一套）", fs=7.2, color=fs.INK,
             ha="left", weight="bold")
    fs.label(ax, 0.040, 0.186, "4 层 transformer，末层用 1 个可学习 query 做交叉注意力；", fs=6.9,
             color=fs.MUTED, ha="left")
    fs.label(ax, 0.040, 0.153, "20 组 head 乘 5 组学习率乘 4 组权重衰减，训 20 轮取最好。", fs=6.9,
             color=fs.MUTED, ha="left")
    fs.label(ax, 0.040, 0.120, "编码器全程冻结，分数只反映表征里有什么。", fs=6.9,
             color=fs.ACCENT, ha="left")

    # ---------- 右：两代的差 ----------
    fs.label(ax, 0.596, 0.952, "(b)　六个分类任务上两代的差", fs=8.8, color=fs.MUTED,
             ha="left", weight="bold")
    bars = [("IN1K", 5.1, fs.GREEN), ("COIN", 4.0, fs.GREEN), ("SSv2", 3.0, fs.BLUE),
            ("K400", 2.8, fs.GREEN), ("Diving-48", 2.3, fs.BLUE), ("Jester", 0.1, fs.BLUE)]
    xb, fac, hh = 0.680, 0.047, 0.048
    ys2 = [0.812, 0.706, 0.600, 0.494, 0.388, 0.282]
    for (name, d, col), y in zip(bars, ys2):
        ax.add_patch(plt.Rectangle((xb, y), d * fac, hh, facecolor=col, edgecolor="none",
                                   zorder=2))
        fs.label(ax, xb - 0.012, y + hh / 2, name, fs=7.2, color=fs.INK, ha="right")
        fs.label(ax, xb + d * fac + 0.010, y + hh / 2, f"+{d:.1f}", fs=7.4, color=col,
                 ha="left", weight="bold")

    ax.plot([xb, xb], [0.238, 0.874], color=fs.FAINT, lw=0.9)
    for t in (0, 2, 4):
        xx = xb + t * fac
        ax.plot([xx, xx], [0.238, 0.232], color=fs.FAINT, lw=0.8)
        fs.label(ax, xx, 0.212, str(t), fs=6.8, color=fs.MUTED)
    fs.label(ax, 0.596, 0.180, "蓝为运动理解，绿为外观理解。", fs=6.9, color=fs.MUTED,
             ha="left")
    fs.label(ax, 0.596, 0.148, "V-JEPA ViT-H 到 V-JEPA 2 ViT-g384，", fs=6.9, color=fs.MUTED,
             ha="left")
    fs.label(ax, 0.596, 0.116, "含数据、参数与分辨率三处同时改动。", fs=6.9, color=fs.MUTED,
             ha="left")

    for name, a, b in [("SSv2", 74.3, 77.3), ("Diving-48", 87.9, 90.2), ("Jester", 97.7, 97.8),
                       ("K400", 84.5, 87.3), ("COIN", 87.1, 91.1), ("IN1K", 80.0, 85.1)]:
        print(f"{name:10s} {a:5.1f} → {b:5.1f}　差 {b - a:+.1f}")
    return fs.save(fig, "practice", "ch06", "eval-matrix")


if __name__ == "__main__":
    fig_tubelet()
    fig_pipeline()
    fig_double_mask()
    fig_video_data()
    fig_eval_matrix()
    # 正文里的词元数按论文的切块方式算出来：时间方向 T/2，每个空间方向 R/16
    tok = {}
    for T, R in ((16, 224), (16, 256), (64, 384)):
        tok[(T, R)] = (T // 2) * (R // 16) ** 2
        print(f"{T} 帧 {R} 像素，2 帧一管、16 像素一步，共 {tok[(T, R)]} 个词元")
    base = tok[(16, 224)]
    print(f"18432 是 1568 的 {tok[(64, 384)] / base:.2f} 倍，注意力矩阵再乘一遍")
    # 规划那一段用到的算术，数来自论文正文
    print(f"800 条样本 × 10 轮迭代 = {800 * 10} 次预测器前向")
    print(f"Cosmos 单个动作 4 分钟 = 240 秒，V-JEPA 2-AC 16 秒，相差 {240 / 16:.0f} 倍")
    # 6.9 数据管线里用到的算术
    mb = 16 * 256 * 256 * 3 / 1e6
    print(f"一段 16 帧 256 x 256 的 uint8 片段 {mb:.2f} MB，2200 万段全存 {mb * 22e6 / 1e6:.0f} TB")
    print(f"16 秒 30 fps 共 {16 * 30} 帧，按 4 fps 抽只要 {16 * 4} 帧，解码量降到 1/{30 / 4:.1f}")
    for steps, clips, name in ((90000, 2e6, "第一代"), (252000, 22e6, "第二代")):
        seen = steps * 3072
        print(f"{name} {steps} 步 x 3072 = {seen / 1e8:.2f} 亿个片段，"
              f"除以 {clips / 1e6:.0f} 百万段是 {seen / clips:.0f} 轮")
    print(f"6.11 的序列长：15 步 x (256 画面 + 动作 + 状态) = {15 * 258} 个词元")
