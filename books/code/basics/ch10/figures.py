# -*- coding: utf-8 -*-
"""第 10 章插图：四种评测协议、标签预算、指标一致性、评测清单。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_protocols():
    fig = plt.figure(figsize=(7.3, 4.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 左侧：冻结编码器
    fs.box(ax, 0.025, 0.100, 0.098, 0.815, "冻结\n编码器\n$E$", fc="#F4F7FA", ec=fs.BLUE,
           fs=8.8, lw=1.3, tc=fs.BLUE)

    # 总线
    ax.plot([0.123, 0.170], [0.508, 0.508], color=fs.INK, lw=1.1, zorder=2)
    ax.plot([0.170, 0.170], [0.185, 0.845], color=fs.INK, lw=1.1, zorder=2)

    lanes = [
        (0.845, "线性探测", ["池化特征\n$z$", "线性层\n$d\\times C$", "类别\n分数"],
         "训练 $dC$ 个参数\n需要全部标签", "accent"),
        (0.625, "$k$ 近邻", ["池化特征\n$z$", "距离排序\n取 $k$ 近邻", "加权\n投票"],
         "不训练\n存整个特征库", "plain"),
        (0.405, "检索", ["池化特征\n$z$", "库内\n全量排序", "Recall\n@k"],
         "不训练\n需要配对目标", "plain"),
        (0.185, "迁移", ["池化特征\n$z$", "注意力头\n或全部解冻", "任务\n精度"],
         "训练头或全部\n每任务存一份", "green"),
    ]

    col = [(0.196, 0.120), (0.328, 0.140), (0.480, 0.112)]
    for ycen, name, boxes, note, kind in lanes:
        h = 0.140
        y0 = ycen - h / 2
        fs.arrow(ax, (0.170, ycen), (0.194, ycen), color=fs.INK, lw=1.0, ms=7)
        fs.label(ax, 0.196, ycen + 0.098, name, fs=8.6, color=fs.INK, ha="left")
        for i, (x, w) in enumerate(col):
            fc, ec, tc = "#FFFFFF", fs.FAINT, fs.INK
            if i == 1:
                if kind == "accent":
                    fc, ec, tc = "#FFF7F0", fs.ACCENT, fs.ACCENT
                elif kind == "green":
                    fc, ec, tc = "#EFFAF4", fs.GREEN, fs.GREEN
            fs.box(ax, x, y0, w, h, boxes[i], fc=fc, ec=ec, tc=tc, fs=7.8, lw=1.1)
            if i < 2:
                fs.arrow(ax, (x + w + 0.002, ycen), (col[i + 1][0] - 0.003, ycen),
                         color=fs.FAINT, lw=0.9, ms=7)
        fs.label(ax, 0.604, ycen, note, fs=7.2, color=fs.MUTED, ha="left")

    fs.label(ax, 0.025, 0.030, "橙色 = 有参数要学　绿色 = 主干可能解冻　白框 = 无参数",
             fs=7.2, color=fs.FAINT, ha="left")
    return fs.save(fig, "basics", "ch10", "protocols")


def fig_label_fraction():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.subplots_adjust(left=0.105, right=0.965, top=0.935, bottom=0.135)

    x = np.array([1, 10, 100])
    scratch = np.array([25.4, 56.4, 78.4])
    pre = np.array([63.0, 74.4])

    ax.axhline(76.5, color=fs.FAINT, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(1.05, 77.4, "线性探测，全部标签：76.5", fontsize=7.4, color=fs.MUTED,
            ha="left", va="bottom")

    ax.plot(x, scratch, color=fs.MUTED, lw=1.6, marker="o", ms=5,
            markerfacecolor="white", markeredgewidth=1.3, zorder=4, label="从零训练")
    ax.plot(x[:2], pre, color=fs.ACCENT, lw=2.0, marker="o", ms=5.6,
            markerfacecolor=fs.ACCENT, zorder=5, label="自监督预训练后微调")

    for xx, yy in zip(x, scratch):
        ax.annotate(f"{yy:.1f}", xy=(xx, yy), xytext=(0, -11), textcoords="offset points",
                    fontsize=7.6, color=fs.MUTED, ha="center")
    for xx, yy in zip(x[:2], pre):
        ax.annotate(f"{yy:.1f}", xy=(xx, yy), xytext=(0, 9), textcoords="offset points",
                    fontsize=7.6, color=fs.ACCENT, ha="center")

    # 两条差距标注
    ax.annotate("", xy=(1.0, 25.4), xytext=(1.0, 63.0),
                arrowprops=dict(arrowstyle="<->", color=fs.BLUE, lw=1.0,
                                shrinkA=1, shrinkB=1))
    ax.text(1.14, 44.0, "差 37.6 个点", fontsize=7.8, color=fs.BLUE, ha="left", va="center")
    ax.annotate("", xy=(10.0, 56.4), xytext=(10.0, 74.4),
                arrowprops=dict(arrowstyle="<->", color=fs.BLUE, lw=1.0,
                                shrinkA=1, shrinkB=1))
    ax.text(10.9, 66.5, "差 18.0", fontsize=7.8, color=fs.BLUE, ha="left", va="center")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["1%", "10%", "100%"])
    ax.set_xlim(0.72, 165)
    ax.set_ylim(15, 87)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_xlabel("可用的 ImageNet 标签比例（对数刻度）", fontsize=8.6, labelpad=4)
    ax.set_ylabel("ImageNet top-1 精度（%）", fontsize=8.6, labelpad=4)
    fs.clean(ax, grid=True, spines=("left", "bottom"))
    ax.legend(loc="lower right", fontsize=8.0, handlelength=1.9,
              labelspacing=0.35, borderpad=0.5)
    ax.text(1.02, 84.2, "1% 与 10% 为微调；100% 处只报线性探测",
            fontsize=7.0, color=fs.FAINT, ha="left", va="center")
    return fs.save(fig, "basics", "ch10", "label-fraction")


def _spearman(a, b):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        return r
    ra, rb = rank(a) - (len(a) + 1) / 2, rank(b) - (len(b) + 1) / 2
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def fig_metric_agreement():
    fig = plt.figure(figsize=(7.3, 3.5))
    axA = fig.add_axes([0.070, 0.150, 0.395, 0.800])
    axB = fig.add_axes([0.585, 0.150, 0.390, 0.800])

    # ---- (a) DINO 表 2：线性探测 vs kNN
    sup = [("监督 RN50", 79.3, 79.3), ("监督 ViT-S", 79.8, 79.8)]
    r50 = [("SCLR", 69.1, 60.7), ("MoCov2", 71.1, 61.9), ("InfoMin", 73.0, 65.3),
           ("BarlowT", 73.2, 66.0), ("OBoW", 73.8, 61.9), ("BYOL", 74.4, 64.8),
           ("DCv2", 75.2, 67.1), ("SwAV", 75.3, 65.7), ("DINO", 75.3, 67.5)]
    vits = [("BYOL", 71.4, 66.6), ("MoCov2", 72.7, 64.4), ("SwAV", 73.5, 66.3),
            ("DINO ViT-S", 77.0, 74.5)]

    lo, hi = 67.5, 81.0
    axA.plot([lo, hi], [lo, hi], color=fs.FAINT, lw=1.0, ls=(0, (4, 3)), zorder=1)
    axA.text(70.4, 71.4, "$y=x$", fontsize=8.0, color=fs.FAINT, ha="left", va="bottom")

    for name, a, b in sup:
        axA.scatter([a], [b], s=52, marker="s", color=fs.INK, edgecolor="white",
                    linewidth=0.9, zorder=5)
    for name, a, b in r50:
        axA.scatter([a], [b], s=40, color=fs.BLUE, edgecolor="white", linewidth=0.8, zorder=4)
    for name, a, b in vits:
        axA.scatter([a], [b], s=46, marker="^", color=fs.GREEN, edgecolor="white",
                    linewidth=0.8, zorder=4)

    axA.annotate("OBoW\n73.8 / 61.9", xy=(73.8, 61.9), xytext=(69.3, 58.9),
                 fontsize=7.2, color=fs.BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=fs.BLUE, lw=0.7,
                                 shrinkA=2, shrinkB=3))
    axA.annotate("DINO ViT-S\n77.0 / 74.5", xy=(77.0, 74.5), xytext=(77.9, 70.2),
                 fontsize=7.2, color=fs.GREEN, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=fs.GREEN, lw=0.7,
                                 shrinkA=2, shrinkB=3))
    axA.text(69.0, 80.2, "监督模型落在线附近", fontsize=7.2, color=fs.INK, ha="left",
             va="center")

    rhoA = _spearman([p[1] for p in sup + r50 + vits], [p[2] for p in sup + r50 + vits])
    axA.text(0.975, 0.045, f"Spearman $\\rho$ = {rhoA:.2f}（15 个模型）",
             transform=axA.transAxes, fontsize=7.4, color=fs.MUTED, ha="right", va="bottom")

    axA.set_xlim(lo, hi); axA.set_ylim(57.0, 82.5)
    axA.set_xticks([68, 70, 72, 74, 76, 78, 80])
    axA.set_yticks([60, 65, 70, 75, 80])
    axA.set_xlabel("线性探测精度（%）", fontsize=8.4, labelpad=3)
    axA.set_ylabel("$k$ 近邻精度（%）", fontsize=8.4, labelpad=2)
    fs.clean(axA, grid=True, spines=("left", "bottom"))
    axA.text(-0.115, 1.035, "(a)", transform=axA.transAxes, fontsize=9.6,
             color=fs.MUTED, weight="bold", ha="left", va="bottom")

    # ---- (b) DINOv2 表 4：线性探测 vs 重新标注后的精度
    weak = [("CLIP L/14", 84.3, 88.1), ("CLIP L/14-336", 85.3, 88.8), ("SWAG H/14", 85.7, 88.7),
            ("OpenCLIP H", 84.4, 88.4), ("OpenCLIP G", 86.2, 89.4), ("EVA-CLIP g", 86.4, 89.3)]
    self_ = [("MAE H/14", 76.6, 83.3), ("DINO S/8", 79.2, 85.5), ("MSN L/7", 80.7, 86.0),
             ("EsViT", 81.3, 87.0), ("Mugs", 82.1, 86.9), ("iBOT L/16", 82.3, 87.5),
             ("DINOv2 S", 81.1, 86.6), ("DINOv2 B", 84.5, 88.3), ("DINOv2 L", 86.3, 89.5),
             ("DINOv2 g", 86.5, 89.6)]

    axB.plot([75, 88], [75, 88], color=fs.FAINT, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for name, a, b in weak:
        axB.scatter([a], [b], s=40, marker="D", color=fs.PURPLE, edgecolor="white",
                    linewidth=0.8, zorder=4)
    for name, a, b in self_:
        axB.scatter([a], [b], s=42, color=fs.BLUE, edgecolor="white", linewidth=0.8, zorder=4)

    axB.annotate("MAE ViT-H/14\n76.6 / 83.3", xy=(76.6, 83.3), xytext=(75.9, 85.4),
                 fontsize=7.2, color=fs.BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=fs.BLUE, lw=0.7,
                                 shrinkA=2, shrinkB=3))
    axB.text(75.4, 90.2, "全部点在 $y=x$ 线上方", fontsize=7.2, color=fs.MUTED, ha="left",
             va="center")

    rhoB = _spearman([p[1] for p in weak + self_], [p[2] for p in weak + self_])
    axB.text(0.030, 0.775, f"Spearman $\\rho$ = {rhoB:.2f}（16 个模型）",
             transform=axB.transAxes, fontsize=7.4, color=fs.MUTED, ha="left", va="bottom")

    axB.set_xlim(75, 88); axB.set_ylim(82, 91)
    axB.set_xticks([76, 78, 80, 82, 84, 86, 88])
    axB.set_yticks([83, 85, 87, 89, 91])
    axB.set_xlabel("线性探测精度（%）", fontsize=8.4, labelpad=3)
    axB.set_ylabel("ReaL（重新标注）上的精度（%）", fontsize=8.4, labelpad=2)
    fs.clean(axB, grid=True, spines=("left", "bottom"))
    axB.text(-0.145, 1.035, "(b)", transform=axB.transAxes, fontsize=9.6,
             color=fs.MUTED, weight="bold", ha="left", va="bottom")

    return fs.save(fig, "basics", "ch10", "metric-agreement")


def fig_checklist():
    fig = plt.figure(figsize=(7.3, 3.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    stages = [
        ("数据", [("1", "说清测什么能力", "分类 / 检索 / 控制"), ("2", "查数据污染", "预训练与测试集重叠")]),
        ("协议", [("3", "冻结边界写全", "BN · dropout · 增强"), ("4", "特征位置与归一化", "层 · token · 池化"),
                  ("5", "超参扫描范围", "选择依据放验证集")]),
        ("指标", [("6", "至少三个种子", "报均值与标准差"), ("7", "至少两种协议", "分歧照实写")]),
        ("结论", [("8", "对齐算力与输入", "参数 · 吞吐 · 分辨率")]),
    ]

    xs = [0.020, 0.270, 0.520, 0.770]
    w = 0.205
    for i, (name, items) in enumerate(stages):
        x = xs[i]
        fs.box(ax, x, 0.845, w, 0.120, name, fc="#F4F7FA", ec=fs.BLUE, tc=fs.BLUE,
               fs=10.0, lw=1.3)
        if i < len(stages) - 1:
            fs.arrow(ax, (x + w + 0.004, 0.905), (xs[i + 1] - 0.005, 0.905),
                     color=fs.MUTED, lw=1.2, ms=9)
        for k, (num, title, sub) in enumerate(items):
            yy = 0.640 - k * 0.185
            h = 0.145
            fs.box(ax, x, yy, w, h, "", fc="#FFFFFF", ec="#D1D5DB", lw=1.0)
            ax.scatter([x + 0.030], [yy + h / 2], s=150, color=fs.ACCENT,
                       edgecolor="none", zorder=6, alpha=0.92)
            ax.text(x + 0.030, yy + h / 2, num, fontsize=7.8, color="white",
                    ha="center", va="center", zorder=7)
            ax.text(x + 0.062, yy + h * 0.66, title, fontsize=8.0, color=fs.INK,
                    ha="left", va="center", zorder=7)
            ax.text(x + 0.062, yy + h * 0.30, sub, fontsize=7.0, color=fs.MUTED,
                    ha="left", va="center", zorder=7)

    fs.label(ax, 0.020, 0.180, "缺任何一条，数字都不可比；换来的是知道结论会被谁、从哪个方向反驳",
             fs=7.8, color=fs.MUTED, ha="left")
    fs.label(ax, 0.020, 0.095, "按阶段推进，前一阶段没过就不要看后一阶段的数字",
             fs=7.8, color=fs.FAINT, ha="left")
    return fs.save(fig, "basics", "ch10", "checklist")


if __name__ == "__main__":
    fig_protocols()
    fig_label_fraction()
    fig_metric_agreement()
    fig_checklist()
