# 同一张世界的地图

### 从语言模型到 JEPA 的三部曲

作者：Dr. Dezeming　·　2026 年 · 第一版

语言模型把整个互联网读了一遍，仍然说不清一个杯子放在桌沿时会不会掉下去。它缺的不是参数，而是关于世界的那份表征。JEPA 给出的答案很克制：不去重建像素，只在表征空间里预测。2026 年的一项工作把这个直觉写成了定理，只要正样本是同一个场景的两次观测，分别训练的模型必然学到同一个表征，只差一个旋转。

这套书分三册。入门篇从输入法联想讲到世界模型，不预设自监督基础；进阶篇进入目标函数、塌缩、掩码与训练配方，面向要动手复现的读者；专业篇给出定理、证明骨架与数值验证，并讨论它在真实数据上管不到的地方。三册共用一套术语与例子，可以按顺序读，也可以从关心的地方切进去。

---

**在线阅读**：<https://ai-crystal.github.io/world-map-jepa/>

| 册 | 副标题 | 页数 | 在线阅读 | PDF |
| --- | --- | --- | --- | --- |
| 01　入门篇 | 从语言模型到 JEPA 的第一课 | 74 | [阅读](books/basics/README.md) | [PDF](output/jepa-basics.pdf) |
| 02　进阶篇 | 联合嵌入预测的训练与工程 | 83 | [阅读](books/practice/README.md) | [PDF](output/jepa-practice.pdf) |
| 03　专业篇 | 柏拉图表征定理与它的边界 | 78 | [阅读](books/theory/README.md) | [PDF](output/jepa-theory.pdf) |

---

## 三册的分工

- **入门篇**回答四个问题：语言模型缺什么、表征是什么、JEPA 为什么在表征空间预测、为什么不同的模型会画出同一张地图。九章读完，不需要任何自监督学习的基础。
- **进阶篇**讲怎么真的训练一个 JEPA：能量的语言、塌缩的解剖、不用负样本的训练、SIGReg、I-JEPA 与 V-JEPA 的装配、视图构造、对齐度量与 42 倍样本效率的算法。每个数字都标了出处与前提。
- **专业篇**把经验和观察换成条件命题：图核与谱嵌入、定理一与定理二、干扰变量的代价、视图与数据共同决定的核、下一词预测的 JEPA 形式，最后回到证据、反驳与边界。

## 仓库结构

```
books/basics/     入门篇（chapters、front、appendix、images）
books/practice/   进阶篇
books/theory/     专业篇
books/code/       生成插图的 Python 代码，按册与章分目录
books/build/      构建脚本与样式
books/common/     共用绘图工具
output/           三册的 PDF
```

正文里的图片用相对路径（`../images/07/xxx.png`），在 GitHub 网页端直接打开章节也能看到图。

## 本地构建

需要 Python 3.10 以上、Chrome 或 Edge。Windows 上双击 `build-basics.bat` 只构建入门篇，`build.bat` 构建全部三册。也可以手动执行：

```bash
python books/build/render_figures.py --book basics
python books/build/build.py --book basics
python books/build/checks/verify_pdf_text.py output/jepa-basics.pdf
```

依赖见 `books/build/requirements.txt`（markdown、playwright、pymupdf、pillow、matplotlib、numpy）。

## 许可

文字与插图采用 CC BY-NC-SA 4.0 许可发布。文中引用的论文、数据集与模型归各自作者所有。
