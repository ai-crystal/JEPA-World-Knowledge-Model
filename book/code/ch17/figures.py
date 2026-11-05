# -*- coding: utf-8 -*-
"""第 17 章插图：旋转对齐的流程、无配对对齐的真实结果、三种对齐程度。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import figstyle as fs


def fig_pipeline():
    """图 17-1：图像编码器 → 旋转 → 文本解码器，只有旋转用了配对。"""
    fig = plt.figure(figsize=(7.0, 3.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------- 上排：训练解码器（只用文本） ----------
    fs.box(ax, 0.010, 0.745, 0.124, 0.125, "训练用的描述", fc="white", ec=fs.MUTED, fs=7.6)
    fs.box(ax, 0.163, 0.730, 0.128, 0.155, "文本 JEPA\n冻结", fc="#F4F7FA", ec=fs.BLUE, fs=8.6)
    fs.box(ax, 0.315, 0.735, 0.126, 0.145, "384 维\n加噪 σ = 3", fc="white", ec=fs.BLUE, fs=8.0)

    fs.arrow(ax, (0.136, 0.808), (0.161, 0.808), color=fs.MUTED, lw=1.1)
    fs.arrow(ax, (0.293, 0.808), (0.313, 0.808), color=fs.MUTED, lw=1.1)

    # 从“384 维加噪”折向解码器
    ax.plot([0.443, 0.778], [0.808, 0.808], color=fs.MUTED, lw=1.1, zorder=2)
    fs.arrow(ax, (0.778, 0.808), (0.778, 0.470), color=fs.MUTED, lw=1.1)

    # ---------- 下排：描述一张新图 ----------
    fs.box(ax, 0.010, 0.300, 0.124, 0.125, "一张图片", fc="white", ec=fs.MUTED, fs=8.2)
    fs.box(ax, 0.163, 0.285, 0.128, 0.155, "DINOv2\n冻结", fc="#F4F7FA", ec=fs.GREEN, fs=8.6)
    fs.box(ax, 0.313, 0.305, 0.100, 0.115, "768 维", fc="white", ec=fs.GREEN, fs=8.4)
    fs.box(ax, 0.443, 0.278, 0.115, 0.170, "旋转 R", fc="#FFF7F0", ec=fs.ACCENT, fs=9.6,
           weight="bold")
    fs.box(ax, 0.588, 0.305, 0.085, 0.115, "384 维", fc="white", ec=fs.GREEN, fs=8.4)
    fs.box(ax, 0.700, 0.245, 0.155, 0.230, "", fc="#F4F7FA", ec=fs.INK, fs=8.6)
    fs.label(ax, 0.7775, 0.415, "文本解码器", fs=8.8, weight="bold")
    fs.label(ax, 0.7775, 0.335, "Qwen2.5-1.5B", fs=7.2, color=fs.MUTED)
    fs.label(ax, 0.7775, 0.292, "或小解码器", fs=7.2, color=fs.MUTED)
    fs.box(ax, 0.878, 0.300, 0.107, 0.125, "一句描述", fc="white", ec=fs.ACCENT, fs=8.4)

    for x0, x1 in ((0.136, 0.161), (0.293, 0.311), (0.415, 0.441),
                   (0.560, 0.586), (0.675, 0.698), (0.857, 0.876)):
        fs.arrow(ax, (x0, 0.3625), (x1, 0.3625), color=fs.MUTED, lw=1.1)

    # ---------- 说明 ----------
    fs.label(ax, 0.500, 0.222, "只在这里用了配对", fs=8.6, color=fs.ACCENT, weight="bold")
    fs.label(ax, 0.7775, 0.196, "从没看过图片", fs=8.4, color=fs.MUTED)
    fs.label(ax, 0.5, 0.085,
             "两个编码器与解码器全部冻结，可训练的参数只有这个 768×384 的旋转",
             fs=8.2, color=fs.MUTED)
    return fs.save(fig, "ch17", "pipeline")


def fig_pairfree():
    """图 17-2：无配对对齐的真实结果（Table 4）。"""
    groups = ["好视图 JEPA 109M", "Qwen3-8B", "CLIP 文本编码器"]
    cols = [fs.ACCENT, fs.BLUE, fs.FAINT]
    panels = [
        ("FOSCTTM ↓ 越低越好", [0.018, 0.018, 0.006], "{:.3f}", (0, 0.026)),
        ("CIFAR-10 零样本", [86.0, 83.4, 84.0], "{:.1f}", (0, 100)),
        ("CIFAR-100 零样本", [67.2, 61.7, 67.0], "{:.1f}", (0, 100)),
        ("ImageNet-100 零样本", [64.4, 47.6, 68.7], "{:.1f}", (0, 100)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.1))
    for ax, (title, vals, fmt, ylim) in zip(axes, panels):
        x = np.arange(3)
        ax.bar(x, vals, width=0.60, color=cols, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (ylim[1] * 0.028), fmt.format(v), ha="center", va="bottom",
                    fontsize=8.4, color=fs.INK)
        ax.set_xticks([])
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9.0, color=fs.INK, pad=7)
        ax.set_yticks([])
        fs.clean(ax, spines=("bottom",))
        ax.tick_params(length=0)
        ax.set_axisbelow(True)
    handles = [Patch(facecolor=c, edgecolor="none") for c in cols]
    fig.legend(handles, groups, loc="lower center", ncol=3, frameon=False,
               fontsize=8.4, handlelength=1.1, handleheight=0.9,
               bbox_to_anchor=(0.5, 0.005), columnspacing=2.2)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.87, bottom=0.17, wspace=0.20)
    return fs.save(fig, "ch17", "pairfree")


def fig_three_align():
    """图 17-3：无对齐、旋转对齐、线性映射对齐（二维示意，几何由脚本算出）。"""
    rng = np.random.default_rng(21)
    n = 22
    x = np.linspace(-1.2, 1.2, n)
    T = np.stack([x, 0.60 * np.sin(1.7 * x + 0.25)]) + 0.02 * rng.normal(size=(2, n))

    # 图像空间的同一批场景：转过去，其中一段还被自己的渲染方式挤了一下
    th = np.deg2rad(28)
    Rtrue = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    bend = 0.9 * np.exp(-((T[0] + 0.62) / 0.45) ** 2)
    I = Rtrue @ (T + np.stack([0.10 * bend, np.zeros(n)])) + 0.02 * rng.normal(size=T.shape)

    idx = np.where(T[0] < -0.2)[0][:6]
    A, B = I[:, idx], T[:, idx]
    M = A @ B.T
    U, S, Vt = np.linalg.svd(M)
    Rhat = Vt.T @ U.T                      # 正交 Procrustes
    rot = Rhat @ I
    Alin = (B @ A.T) @ np.linalg.inv(A @ A.T)   # 一般线性映射
    lin = Alin @ I

    fitted = np.zeros(n, bool)
    fitted[idx] = True

    allp = np.concatenate([T, I, rot, lin], axis=1)
    x0, x1 = allp[0].min() - 0.16, allp[0].max() + 0.16
    y0, y1 = allp[1].min() - 0.16, allp[1].max() + 0.16

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.85))
    sets = [(I, "(a) 无对齐", "Flickr .043　COCO .038"),
            (rot, "(b) 旋转对齐", "Flickr .538　COCO .832"),
            (lin, "(c) 线性映射对齐", "Flickr .561　COCO .652")]

    for k, (ax, (P, title, score)) in enumerate(zip(axes, sets)):
        for j in range(n):
            if k == 2 and fitted[j]:
                ax.plot([P[0, j], T[0, j]], [P[1, j], T[1, j]], color=fs.ACCENT,
                        lw=1.5, alpha=0.9, zorder=2)
            else:
                ax.plot([P[0, j], T[0, j]], [P[1, j], T[1, j]], color=fs.FAINT,
                        lw=0.55, alpha=0.6, zorder=2)
        ax.scatter(T[0], T[1], s=15, facecolor="white", edgecolor=fs.MUTED,
                   linewidth=0.9, zorder=4)
        if k == 2:
            ax.scatter(P[0, ~fitted], P[1, ~fitted], s=14, color=fs.BLUE, zorder=5)
            ax.scatter(P[0, fitted], P[1, fitted], s=24, color=fs.ACCENT,
                       marker="s", zorder=6)
        else:
            ax.scatter(P[0], P[1], s=14, color=fs.BLUE, zorder=5)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=9.4, color=fs.INK, pad=6)
        ax.text(0.5, -0.04, score, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, color=fs.MUTED)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.13, wspace=0.05)
    return fs.save(fig, "ch17", "three-align")


if __name__ == "__main__":
    fig_pipeline()
    fig_pairfree()
    fig_three_align()

    # 章里引用的两个参数量，由本脚本算出
    for da, db in ((768, 384), (5120, 384)):
        print(f"{da}x{db}: {da * db - db * (db + 1) // 2:,} 个参数")
    print("384 维旋转自由度", 384 * 383 // 2, "全矩阵", 384 * 384)
