# -*- coding: utf-8 -*-
"""第 11 章插图：CLIP 双塔、VLM 三阶段、图像离散化、图文覆盖缺口。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle, Polygon, FancyArrowPatch

import figstyle as fs

BOOK = "basics"
CH = "ch11"


# ---------------------------------------------------------------- 图 11-1
def fig_clip_towers():
    fig, ax = fs.canvas(7.2, 4.35)

    # 图像塔
    fs.box(ax, 0.015, 0.745, 0.072, 0.140, "图像\n$x$", fc=fs.LIGHT, ec=fs.FAINT, fs=8.2, lw=1.0)
    fs.arrow(ax, (0.090, 0.815), (0.112, 0.815), color=fs.MUTED, lw=1.0)
    fs.box(ax, 0.114, 0.745, 0.150, 0.140, "图像编码器\nViT-L/14",
           fc="#FFFFFF", ec=fs.BLUE, tc=fs.BLUE, fs=7.8, lw=1.2)
    fs.arrow(ax, (0.267, 0.815), (0.289, 0.815), color=fs.MUTED, lw=1.0)
    fs.box(ax, 0.291, 0.745, 0.082, 0.140, "线性投影\n$W_i$", fc="#FFFFFF", ec=fs.BLUE,
           tc=fs.BLUE, fs=7.8, lw=1.2)
    fs.arrow(ax, (0.376, 0.815), (0.398, 0.815), color=fs.MUTED, lw=1.0)
    fs.box(ax, 0.400, 0.745, 0.100, 0.140, "$u_i$\n单位向量", fc="#EFF6FF", ec=fs.BLUE,
           tc=fs.BLUE, fs=7.8, lw=1.1)

    # 文本塔
    fs.box(ax, 0.015, 0.525, 0.072, 0.140, "文本\n$y$", fc=fs.LIGHT, ec=fs.FAINT, fs=8.2, lw=1.0)
    fs.arrow(ax, (0.090, 0.595), (0.112, 0.595), color=fs.MUTED, lw=1.0)
    fs.box(ax, 0.114, 0.525, 0.150, 0.140, "文本编码器\nTransformer",
           fc="#FFFFFF", ec=fs.GREEN, tc=fs.GREEN, fs=7.8, lw=1.2)
    fs.arrow(ax, (0.267, 0.595), (0.289, 0.595), color=fs.MUTED, lw=1.0)
    fs.box(ax, 0.291, 0.525, 0.082, 0.140, "线性投影\n$W_t$", fc="#FFFFFF", ec=fs.GREEN,
           tc=fs.GREEN, fs=7.8, lw=1.2)
    fs.arrow(ax, (0.376, 0.595), (0.398, 0.595), color=fs.MUTED, lw=1.0)
    fs.box(ax, 0.400, 0.525, 0.100, 0.140, "$v_j$\n单位向量", fc="#ECFDF5", ec=fs.GREEN,
           tc=fs.GREEN, fs=7.8, lw=1.1)

    # 相似度矩阵
    x0, y0, cw, chh, n = 0.610, 0.440, 0.0355, 0.0470, 8
    for r in range(n):
        for c in range(n):
            diag = (r == c)
            ax.add_patch(Rectangle((x0 + c * cw, y0 + (n - 1 - r) * chh), cw, chh,
                                   facecolor="#FFEDD5" if diag else "#F8FAFC",
                                   edgecolor=fs.FAINT, lw=0.5, zorder=3))
    fs.arrow(ax, (0.505, 0.782), (0.604, 0.742), color=fs.BLUE, lw=1.1, ls=(0, (3, 2)))
    fs.arrow(ax, (0.505, 0.575), (0.604, 0.590), color=fs.GREEN, lw=1.1, ls=(0, (3, 2)))
    fs.label(ax, 0.752, 0.885, "相似度矩阵，$N\\times N$", fs=7.8, color=fs.INK)
    fs.label(ax, 0.610, 0.375, "行是图，列是文，$s_{ij}=u_i^{\\top}v_j/\\tau$",
             fs=7.4, color=fs.INK, ha="left")
    fs.label(ax, 0.610, 0.325, "橙色是对角线上的正样本", fs=7.4, color=fs.ACCENT, ha="left")
    fs.label(ax, 0.610, 0.275, "两个方向各做一次 softmax", fs=7.4, color=fs.MUTED, ha="left")

    # 左侧注记
    notes = ["四亿图文对，批 32,768，训 32 轮",
             "增强只用随机方形裁剪",
             "温度 $\\tau$ 从 0.07 起步，随训练一起学",
             "零样本 ImageNet 76.2%，三十多个数据集同测"]
    for i, t in enumerate(notes):
        fs.label(ax, 0.015, 0.240 - i * 0.052, t, fs=7.6, color=fs.MUTED, ha="left")

    fs.label(ax, 0.015, 0.940, "图像塔", fs=8.6, color=fs.BLUE, ha="left", weight="bold")
    fs.label(ax, 0.015, 0.470, "文本塔", fs=8.6, color=fs.GREEN, ha="left", weight="bold")

    fs.save(fig, BOOK, CH, "clip-towers")


# ---------------------------------------------------------------- 图 11-2
def fig_vlm_stages():
    fig, ax = fs.canvas(7.2, 3.85)

    panels = [
        (0.030, "(a) 对齐", "只训投影层", 
         {"vision": ("视觉编码器", True), "proj": ("投影层", False), "llm": ("语言模型", True)},
         "数据：图文对"),
        (0.372, "(b) 指令微调", "训投影层与语言模型",
         {"vision": ("视觉编码器", True), "proj": ("投影层", False), "llm": ("语言模型", False)},
         "数据：问答与对话"),
        (0.714, "(c) 多模态继续训练", "解冻视觉主干",
         {"vision": ("视觉编码器", False), "proj": ("投影层", False), "llm": ("语言模型", False)},
         "数据：混合语料"),
    ]

    for x0, title, sub, spec, dat in panels:
        fs.label(ax, x0 + 0.13, 0.930, title, fs=9.0, color=fs.INK, weight="bold")
        fs.label(ax, x0 + 0.13, 0.868, sub, fs=7.8, color=fs.MUTED)

        ys = {"vision": 0.640, "proj": 0.430, "llm": 0.220}
        for key in ("vision", "proj", "llm"):
            name, frozen = spec[key]
            if frozen:
                fs.box(ax, x0 + 0.020, ys[key], 0.220, 0.150, name + "（冻结）",
                       fc=fs.LIGHT, ec=fs.FAINT, tc=fs.MUTED, fs=8.0, lw=1.0)
                ax.add_patch(Rectangle((x0 + 0.020, ys[key]), 0.220, 0.150,
                                       fill=False, ec=fs.FAINT, lw=0.9,
                                       linestyle=(0, (3, 2)), zorder=4))
            else:
                fs.box(ax, x0 + 0.020, ys[key], 0.220, 0.150, name,
                       fc="#FFF7ED", ec=fs.ACCENT, tc=fs.ACCENT, fs=8.2, lw=1.4)

        fs.arrow(ax, (x0 + 0.130, 0.636), (x0 + 0.130, 0.586), color=fs.MUTED, lw=1.0)
        fs.arrow(ax, (x0 + 0.130, 0.426), (x0 + 0.130, 0.376), color=fs.MUTED, lw=1.0)
        fs.label(ax, x0 + 0.130, 0.150, dat, fs=7.6, color=fs.MUTED)

    # 图例
    ax.add_patch(Rectangle((0.030, 0.038), 0.016, 0.036, facecolor=fs.LIGHT,
                           edgecolor=fs.FAINT, lw=0.9, zorder=4))
    fs.label(ax, 0.052, 0.056, "冻结", fs=7.6, color=fs.MUTED, ha="left")
    ax.add_patch(Rectangle((0.108, 0.038), 0.016, 0.036, facecolor="#FFF7ED",
                           edgecolor=fs.ACCENT, lw=1.2, zorder=4))
    fs.label(ax, 0.130, 0.056, "可训练", fs=7.6, color=fs.ACCENT, ha="left")
    fs.label(ax, 0.300, 0.056, "三段之间改动的参数越来越多，需要的数据也越来越多",
             fs=7.6, color=fs.MUTED, ha="left")

    fs.save(fig, BOOK, CH, "vlm-stages")


# ---------------------------------------------------------------- 图 11-3
def fig_vq_tokenizer():
    fig, ax = fs.canvas(7.2, 4.30)

    # 原图
    fs.box(ax, 0.015, 0.660, 0.068, 0.210, "图像\n$128^2$", fc=fs.LIGHT, ec=fs.FAINT,
           fs=7.8, lw=1.0)
    fs.arrow(ax, (0.085, 0.765), (0.101, 0.765), color=fs.MUTED, lw=1.0)

    # 编码器
    fs.box(ax, 0.103, 0.660, 0.072, 0.210, "编码器\n$z_e$", fc="#FFFFFF", ec=fs.BLUE,
           tc=fs.BLUE, fs=7.8, lw=1.2)
    fs.arrow(ax, (0.177, 0.765), (0.193, 0.765), color=fs.MUTED, lw=1.0)

    # 潜在网格
    gx, gy, gw, gh, n = 0.195, 0.675, 0.0225, 0.0265, 4
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((gx + c * gw, gy + r * gh), gw, gh,
                                   facecolor="#DBEAFE", edgecolor=fs.BLUE, lw=0.5, zorder=3))
    fs.label(ax, 0.240, 0.912, "连续潜变量 $z_e$", fs=7.6, color=fs.BLUE)

    # 最近邻
    fs.arrow(ax, (0.290, 0.730), (0.316, 0.730), color=fs.ACCENT, lw=1.2)


    # 码本
    cbx, cby, cbw, cbh = 0.320, 0.610, 0.108, 0.310
    fs.box(ax, cbx, cby, cbw, cbh, "", fc="#FFFFFF", ec=fs.PURPLE, lw=1.1)
    fs.label(ax, cbx + cbw / 2, cby + cbh + 0.034, "码本 $K=512$，最近邻查找", fs=7.2,
             color=fs.PURPLE)
    for i in range(6):
        yy = cby + cbh - 0.048 - i * 0.047
        if i == 2:
            ax.add_patch(Rectangle((cbx + 0.008, yy - 0.018), cbw - 0.016, 0.038,
                                   facecolor="#F3E8FF", edgecolor="none", zorder=4))
        fs.label(ax, cbx + 0.014, yy, f"$e_{{{i + 1}}}$", fs=7.2, color=fs.PURPLE, ha="left")
        dots = np.linspace(cbx + 0.038, cbx + cbw - 0.014, 5)
        for d in dots:
            ax.plot([d, d], [yy - 0.012, yy + 0.012], color=fs.FAINT, lw=1.0, zorder=5)
    fs.label(ax, cbx + 0.030, cby + 0.018, "$\\vdots$", fs=7.4, color=fs.PURPLE, ha="left")

    fs.arrow(ax, (0.430, 0.730), (0.446, 0.730), color=fs.MUTED, lw=1.0)

    # 索引网格
    ix, iy, giw = 0.446, 0.675, 0.0245
    rng = np.random.default_rng(3)
    ids = rng.integers(100, 512, size=(n, n))
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((ix + c * giw, iy + r * gh), giw, gh,
                                   facecolor="#F5F3FF", edgecolor=fs.PURPLE, lw=0.5, zorder=3))
            fs.label(ax, ix + c * giw + giw / 2, iy + r * gh + gh / 2, str(ids[r, c]),
                     fs=4.4, color=fs.PURPLE)
    fs.label(ax, 0.495, 0.655, "索引网格", fs=6.6, color=fs.PURPLE)
    fs.label(ax, 0.495, 0.618, "1 到 512", fs=6.6, color=fs.PURPLE)

    fs.arrow(ax, (0.545, 0.765), (0.557, 0.765), color=fs.MUTED, lw=1.0)

    # 解码器
    fs.box(ax, 0.559, 0.660, 0.070, 0.210, "解码器", fc="#FFFFFF", ec=fs.BLUE,
           tc=fs.BLUE, fs=7.8, lw=1.2)
    fs.arrow(ax, (0.631, 0.765), (0.647, 0.765), color=fs.MUTED, lw=1.0)
    fs.box(ax, 0.649, 0.660, 0.068, 0.210, "重建\n$128^2$", fc=fs.LIGHT, ec=fs.FAINT,
           fs=7.8, lw=1.0)

    # 右侧注记
    fs.label(ax, 0.735, 0.870, "128×128×3 → 32×32×1", fs=7.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.735, 0.815, "每个码字 64 维", fs=7.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.735, 0.760, "按比特算压约 42.6 倍", fs=7.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.735, 0.705, "重建只比原图糊一点", fs=7.4, color=fs.MUTED, ha="left")
    # 梯度直通回路
    ax.plot([0.594, 0.594], [0.658, 0.606], color=fs.ACCENT, lw=0.9, ls=(0, (3, 2)))
    ax.plot([0.594, 0.177], [0.606, 0.606], color=fs.ACCENT, lw=0.9, ls=(0, (3, 2)))
    ax.add_patch(FancyArrowPatch((0.177, 0.606), (0.177, 0.656), arrowstyle="-|>",
                                 mutation_scale=7, color=fs.ACCENT, lw=0.9,
                                 linestyle=(0, (3, 2)), zorder=4))
    fs.label(ax, 0.240, 0.575, "梯度直通：解码器输入端算出的梯度", fs=6.9,
             color=fs.ACCENT, ha="left")
    fs.label(ax, 0.240, 0.536, "原样抄给编码器输出，绕过取整这一步", fs=6.9,
             color=fs.ACCENT, ha="left")

    # 三项损失
    fs.label(ax, 0.500, 0.489, "总损失是三段之和，各段只对一部分参数生效",
             fs=7.6, color=fs.INK)
    losses = [
        (0.020, "重建损失", "$\\Vert x-\\hat x\\Vert^2$", "训练编码器与解码器"),
        (0.350, "码本损失", "$\\Vert \\mathrm{sg}[z_e]-e_k\\Vert^2$", "只移动被选中的码字"),
        (0.680, "承诺损失", "$\\beta\\Vert z_e-\\mathrm{sg}[e_k]\\Vert^2$", "把编码器输出往码字上拉，$\\beta=0.25$"),
    ]
    for x0, t1, t2, t3 in losses:
        fs.box(ax, x0, 0.150, 0.300, 0.300, "", fc="#FFFFFF", ec=fs.FAINT, lw=1.0)
        fs.label(ax, x0 + 0.150, 0.395, t1, fs=8.2, color=fs.INK, weight="bold")
        fs.label(ax, x0 + 0.150, 0.310, t2, fs=8.0, color=fs.ACCENT)
        fs.label(ax, x0 + 0.150, 0.220, t3, fs=7.2, color=fs.MUTED)

    fs.save(fig, BOOK, CH, "vq-tokenizer")


# ---------------------------------------------------------------- 图 11-4
def fig_caption_gap():
    fig, ax = fs.canvas(7.2, 3.95)

    # 画面边框
    L, R, B, T = 0.030, 0.330, 0.090, 0.920
    ax.add_patch(Rectangle((L, B), R - L, T - B, facecolor="#FCFCFD",
                           edgecolor=fs.FAINT, lw=1.0, zorder=1))
    ax.add_patch(Rectangle((L, B), R - L, 0.250, facecolor="#E8F0E3",
                           edgecolor="none", zorder=2))
    for i in range(9):
        xx = L + 0.012 + i * 0.033
        ax.plot([xx, xx + 0.012], [B + 0.020 + 0.026 * (i % 3), B + 0.062 + 0.026 * (i % 3)],
                color="#CFDCC5", lw=0.9, zorder=3)
    # 太阳
    ax.add_patch(Circle((0.283, 0.862), 0.015, facecolor="#F6D98A",
                        edgecolor="#D9AE4E", lw=0.8, zorder=4))
    for k in range(8):
        th = 2 * np.pi * k / 8
        ax.plot([0.283 + 0.021 * np.cos(th), 0.283 + 0.030 * np.cos(th)],
                [0.862 + 0.021 * np.sin(th), 0.862 + 0.030 * np.sin(th)],
                color="#D9AE4E", lw=0.8, zorder=4)
    # 影子
    ax.add_patch(Ellipse((0.168, 0.316), 0.158, 0.034, facecolor="#DCE3D8",
                         edgecolor="none", zorder=3))
    # 狗身、头、耳、腿、尾
    ax.add_patch(Ellipse((0.163, 0.472), 0.128, 0.104, facecolor="#D9C3A9",
                         edgecolor=fs.INK, lw=0.9, zorder=5))
    ax.add_patch(Ellipse((0.238, 0.516), 0.064, 0.066, facecolor="#D9C3A9",
                         edgecolor=fs.INK, lw=0.9, zorder=6))
    ax.add_patch(Polygon([(0.224, 0.550), (0.235, 0.588), (0.248, 0.550)],
                         closed=True, facecolor="#B99B7B", edgecolor=fs.INK, lw=0.7, zorder=6))
    ax.plot([0.235, 0.232], [0.532, 0.548], color=fs.INK, lw=1.0, zorder=7)
    ax.plot([0.262, 0.270], [0.506, 0.502], color=fs.INK, lw=0.9, zorder=7)
    for dx, y1 in ((0.120, 0.330), (0.152, 0.330), (0.186, 0.330), (0.214, 0.330)):
        ax.plot([dx, dx], [0.428, y1], color=fs.INK, lw=2.4, solid_capstyle="round", zorder=4)
    ax.plot([0.104, 0.090, 0.092], [0.492, 0.516, 0.540], color=fs.INK, lw=2.0,
            solid_capstyle="round", zorder=4)
    # 飞盘
    ax.add_patch(Ellipse((0.290, 0.640), 0.060, 0.026, angle=-18,
                         facecolor="#F2C14E", edgecolor=fs.INK, lw=0.9, zorder=6))
    # 右下角半只脚
    ax.add_patch(Ellipse((0.312, 0.132), 0.060, 0.030, facecolor="#8B5E3C",
                         edgecolor=fs.INK, lw=0.8, zorder=6, alpha=0.9))

    for x, y, t in [(0.278, 0.700, "④"), (0.148, 0.292, "⑤"), (0.300, 0.186, "⑥")]:
        ax.add_patch(Circle((x, y), 0.016, facecolor="white", edgecolor=fs.ACCENT,
                            lw=0.9, zorder=8))
        fs.label(ax, x, y - 0.001, t, fs=6.2, color=fs.ACCENT, z=9)

    # 配文
    fs.box(ax, 0.355, 0.845, 0.215, 0.075, "“一只狗在草地上接飞盘”",
           fc="#FFFFFF", ec=fs.GREEN, tc=fs.GREEN, fs=7.2, lw=1.1, rounded=0.015)
    fs.label(ax, 0.356, 0.798, "网页上配的一句话", fs=7.0, color=fs.MUTED, ha="left")

    fs.label(ax, 0.355, 0.735, "文字写到的", fs=8.4, color=fs.GREEN, ha="left", weight="bold")
    for i, t in enumerate(["√ 主体是一条狗", "√ 有一条飞盘", "√ 场景是草地"]):
        fs.label(ax, 0.355, 0.668 - i * 0.062, t, fs=7.6, color=fs.INK, ha="left")

    fs.label(ax, 0.600, 0.735, "图里还有、文字没说的", fs=8.4, color=fs.ACCENT,
             ha="left", weight="bold")
    right = ["④ 飞盘在狗嘴前上方，还没碰到", "⑤ 光从右边来，影子朝左",
             "⑥ 右下角露出半只人脚", "草被踩倒一片，说明刚跑过",
             "狗耳朵向后压，身体前倾"]
    for i, t in enumerate(right):
        fs.label(ax, 0.600, 0.668 - i * 0.062, t, fs=7.6, color=fs.INK, ha="left")

    fs.label(ax, 0.600, 0.318, "这五条都能从图里读出来，", fs=7.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.600, 0.268, "但没有任何一条会写进配文，", fs=7.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.600, 0.218, "因为写配文的人默认读者自己看得见。", fs=7.4, color=fs.MUTED, ha="left")
    fs.label(ax, 0.355, 0.140, "示意图，不对应统计测量", fs=7.0, color=fs.FAINT, ha="left")

    fs.save(fig, BOOK, CH, "caption-gap")


if __name__ == "__main__":
    fig_clip_towers()
    fig_vlm_stages()
    fig_vq_tokenizer()
    fig_caption_gap()
