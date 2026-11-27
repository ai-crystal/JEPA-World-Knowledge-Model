# 同一张世界的地图

> 从语言模型到 JEPA 的三部曲

语言模型把整个互联网读了一遍，仍然说不清一个杯子放在桌沿时会不会掉下去。它缺的不是参数，而是关于世界的那份表征。JEPA 给出的答案很克制：不去重建像素，只在表征空间里预测。

这套书分三册。入门篇从输入法联想讲到世界模型；进阶篇进入目标函数、塌缩、掩码与训练配方；专业篇给出定理、证明骨架与数值验证。三册共用一套术语与例子。

作者：Dr. Dezeming　·　2026

**在线阅读**：[在线阅读站](https://ai-crystal.github.io/JEPA-World-Knowledge-Model/)　·　**仓库**：[GitHub 仓库](https://github.com/ai-crystal/JEPA-World-Knowledge-Model)

## 分册

| 册 | 书名 | 一句话 | 页数 | 在线阅读 | PDF |
| --- | --- | --- | ---: | --- | --- |
| 01 | [入门篇](books/basics/README.md) | 从语言模型到 JEPA 的第一课 | 147 | [阅读](books/basics/README.md) | <a href="output/v1-basics.pdf" target="_blank" rel="noopener" data-router-ignore>下载</a> |
| 02 | [进阶篇](books/practice/README.md) | 联合嵌入预测的训练与工程 | 141 | [阅读](books/practice/README.md) | <a href="output/v2-practice.pdf" target="_blank" rel="noopener" data-router-ignore>下载</a> |
| 03 | [专业篇](books/theory/README.md) | 柏拉图表征定理与它的边界 | 128 | [阅读](books/theory/README.md) | <a href="output/v3-theory.pdf" target="_blank" rel="noopener" data-router-ignore>下载</a> |

## 怎么读

- 第一次接触 JEPA：读入门篇，不需要任何自监督学习的基础。
- 要动手复现：读进阶篇，重点是塌缩、SIGReg、视图构造与训练配方。
- 关心定理与边界：读专业篇，重点是图核、谱嵌入、定理一与定理二。

## 许可

本系列文字与插图采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可发布，转载请注明作者与出处。书中引用的论文、数据集与模型归各自作者所有。

---

作者：Dr. Dezeming　·　2026
