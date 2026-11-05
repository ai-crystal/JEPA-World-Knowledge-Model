# 同一张世界的地图

### 从语言模型到 JEPA 的世界模型

作者：Dr. Dezeming　·　2026 年 · 第一版

语言模型把整个互联网读了一遍，仍然说不清一个杯子放在桌沿时会不会掉下去。它缺的不是参数，而是关于世界的那份表征。

JEPA 给出的答案很克制：不去重建像素，只在表征空间里预测。2026 年的一项工作把这个直觉写成了定理，只要正样本是同一个场景的两次观测，分别训练的模型必然学到同一个表征，只差一个旋转。

本书从下一个词预测讲起，经过表征、能量模型、塌缩、I-JEPA、V-JEPA 与世界模型，最后落到那个定理，以及它解释不了的部分。

---

**在线阅读**：<https://ai-crystal.github.io/world-map-jepa/>

**下载 PDF**：[output/world-map-jepa.pdf](output/world-map-jepa.pdf)（142 页，适合打印与离线阅读）

---

## 目录

### 第一篇　语言模型的边界

- [第 1 章　预测下一个词](book/chapters/ch01.md)
- [第 2 章　表征是把世界折起来](book/chapters/ch02.md)
- [第 3 章　三条自监督路线](book/chapters/ch03.md)

### 第二篇　联合嵌入预测

- [第 4 章　用能量衡量兼容](book/chapters/ch04.md)
- [第 5 章　塌缩](book/chapters/ch05.md)
- [第 6 章　不用负样本的学习](book/chapters/ch06.md)
- [第 7 章　把表征捏成高斯](book/chapters/ch07.md)
- [第 8 章　在表征空间里预测](book/chapters/ch08.md)
- [第 9 章　从图像到视频](book/chapters/ch09.md)

### 第三篇　世界模型

- [第 10 章　世界模型](book/chapters/ch10.md)
- [第 11 章　让 JEPA 成为世界模型](book/chapters/ch11.md)

### 第四篇　柏拉图式的表征

- [第 12 章　趋同的证据](book/chapters/ch12.md)
- [第 13 章　三种解释与一场争论](book/chapters/ch13.md)
- [第 14 章　从假设到定理](book/chapters/ch14.md)
- [第 15 章　好视图与坏视图](book/chapters/ch15.md)

### 第五篇　证据与边界

- [第 16 章　四十二倍的差距](book/chapters/ch16.md)
- [第 17 章　一个旋转能做什么](book/chapters/ch17.md)

### 附录

- [附录 A　数学补充](book/appendix/appA.md)
- [附录 B　术语表](book/appendix/appB.md)
- [附录 C　参考文献](book/appendix/appC.md)

- [后记](book/front/afterword.md)

## 仓库结构

```
book/chapters/   正文章节（唯一一份书稿）
book/front/      前言、篇首、后记
book/appendix/   附录
book/images/     插图，按章分目录
book/code/       生成插图的 Python 代码
book/build/      PDF 构建脚本与样式
output/          成书 PDF
```

正文里的图片用相对路径（`../images/07/xxx.png`），在 GitHub 网页端直接打开章节也能看到图。

## 本地构建

需要 Python 3.10 以上、Chrome 或 Edge。Windows 上双击根目录的 `一键生成PDF.bat` 即可，
它会依次重绘插图、编译 PDF、检查排版。也可以手动执行：

```bash
python book/build/render_figures.py
python book/build/build_book.py
python book/build/checks/verify_pdf_text.py
```

依赖见 `book/build/requirements.txt`（markdown、playwright、pymupdf、pillow、matplotlib、numpy）。

## 许可

文字与插图采用 CC BY-NC-SA 4.0 许可发布。文中引用的论文、数据集与模型归各自作者所有。
