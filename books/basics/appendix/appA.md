# 附录 A　术语与延伸阅读

## A.1　术语表

下表按中文名的拼音排序，英文写在括号里。读正文时遇到想不起来的词，可以先在这里花十秒确认它指什么，再决定要不要翻回原处。每条只给一句够用的解释，不负责把来龙去脉讲全。

| 术语（英文） | 一句话解释 |
| --- | --- |
| 表征（representation） | 把对象映射成向量，保留对下游有用的差异，丢掉无关的差异 |
| 柏拉图式表征假设（PRH） | 猜想不同模型在变大变强时收敛到同一个关于世界的统计模型 |
| 对比学习（contrastive learning） | 拉近正样本、推远负样本的自监督方法，需要大量负样本 |
| 冻结编码器（frozen encoder） | 评测时不再更新权重的编码器，用来单独考察表征本身 |
| 负样本（negative sample） | 被明确要求远离锚点的样本，用来防止所有表征挤在一起 |
| 各向同性高斯（isotropic Gaussian） | 每个方向方差相同的高斯分布，LeJEPA 指定的目标分布 |
| 互近邻（mutual k-NN） | 同一批样本在两个表征空间里近邻集合的重合比例，不需要拟合任何映射 |
| 汇聚时域控制（MPC） | 每次只执行计划的开头一小段，然后用真实观测重新规划 |
| 交叉熵（cross-entropy） | 给正确答案分配的概率取负对数，语言模型的训练损失 |
| 联合嵌入预测架构（JEPA） | 先把两种观察各自压成向量，再让一个网络从一个向量预测另一个 |
| 能量模型（energy-based model） | 用一个标量衡量输入与输出有多兼容，推理就是找能量低处 |
| 配分函数（partition function） | 把所有状态的能量指数加起来的归一化常数，通常算不出来也用不着算 |
| 嵌入（embedding） | 表征的另一种叫法，强调它落在某个向量空间里 |
| 视图（view） | 同一个样本经过变换得到的另一个观测，正样本由视图对构成 |
| 世界模型（world model） | 给定状态和动作预测下一个状态，并能在内部连续推演的模型 |
| 塌缩（collapse） | 编码器把所有输入映射成同一个向量或同一个低维子空间 |
| 停梯度（stop-gradient） | 反向传播时切断某一支的梯度，防止两支一起塌缩 |
| 投影头（projection head） | 接在编码器后面的小网络，只在训练时使用，评测时丢掉 |
| 线性探测（linear probe） | 冻结编码器后只训练一层线性分类器，衡量表征的线性可分性 |
| 信息噪声对比估计（InfoNCE） | 对比学习的标准损失，把正样本从一批候选里认出来 |
| 掩码建模（masked modeling） | 遮住输入的一部分，让模型预测被遮住的内容 |
| 指数滑动平均（EMA） | 目标网络的参数按固定比例缓慢跟随在线网络，不接收梯度 |
| 自监督学习（self-supervised learning） | 从数据自身构造监督信号，不需要人工标注 |
| 正样本（positive sample） | 被声明为“应该相似”的一对观测，JEPA 的全部结论都建立在它上面 |
| 注意力（attention） | 让每个位置对其他位置加权求和，Transformer 的核心运算 |

## A.2　延伸阅读

这一册涉及的工作按主题列在下面。想读原文的读者可以直接按 arXiv 编号检索，中文综述可以配合进阶篇与专业篇一起看。

- 语言模型与规模：Brown 等，*Language Models are Few-Shot Learners*，2020（arXiv:2005.14165）；Kaplan 等，*Scaling Laws for Neural Language Models*，2020（arXiv:2001.08361）。
- 注意力与视觉主干：Vaswani 等，*Attention Is All You Need*，2017（arXiv:1706.03762）；Dosovitskiy 等，*An Image is Worth 16x16 Words*，2021（arXiv:2010.11929）。
- 掩码与对比：He 等，*Masked Autoencoders Are Scalable Vision Learners*，2022（arXiv:2111.06377）；Chen 等，*A Simple Framework for Contrastive Learning*，2020（arXiv:2002.05709）；He 等，*Momentum Contrast*，2020（arXiv:1911.05722）。
- JEPA 家族：Assran 等，*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*，2023（arXiv:2301.08243）；Bardes 等，*Revisiting Feature Prediction for Learning Visual Representations from Video*，2024（arXiv:2404.08471）；Assran 等，*V-JEPA 2*，2025（arXiv:2506.09985）。
- 世界模型：Ha 与 Schmidhuber，*World Models*，2018（arXiv:1803.10122）；Hafner 等，*Mastering Diverse Domains through World Models*，2023（arXiv:2301.04104）。
- 表征趋同：Huh 等，*The Platonic Representation Hypothesis*，2024（arXiv:2405.07987）。
