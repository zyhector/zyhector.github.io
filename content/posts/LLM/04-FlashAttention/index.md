---
title: "「从零开始学大模型」Flash Attention"
subtitle: ""
date: 2026-03-18T17:01:19-07:00
draft: false
toc:
    enable: true
weight: false
categories: ["人工智能"]
tags: ["LLM", "大模型", "Transformer", "笔记", "算法", "Flash Attention"]
---

Flash Attention针对的痛点是传统 Transformer 在 decode 阶段，计算 attention 时的巨大内存瓶颈。

因为 Attention Block 需要 `MatMul -> Mask -> Softmax -> Dropout -> Matmul` 这么多步骤，会反复的将大矩阵从 HBM 里搬入 SRAM，计算，再搬出，造成了内存瓶颈。

## 核心概念

Flash Attention 有两个核心 idea 来解决这个痛点：

- 算子融合：它将 `MatMul -> Mask -> Softmax -> Dropout -> Matmul` 这个算子调用链，进行融合。让中间数据不写回显存，也不用重新从显存里读取，全部使用高速的 SRAM 存中间结果。
- 分块 Tiling：因为大矩阵不能全部放进 SRAM，它需要将大矩阵乘法拆解，变成只针对少量数据的计算。

因此说：

> “FlashAttention: **Fast** and **Memory-Efficient** **Exact** Attention with **IO-Awareness**”

- Fast: 快！
- Memory-Efficient：传统 attention 是 $O(N^2)$ 的，这个则是sub-quadratic/linear的 $N(O(N))$ 的
- Exact：没有通过精度损失来实现速度，就是精确的 attention 值
- IO-Awareness：通过参数自动计算 Tile，实现“适配内存”

## Softmax 难题

想要将 Attention 的计算进行分块处理，唯一难啃的是 Softmax。

矩阵乘法（$QK^T$ 和 $PV$）天然就支持分块计算（Block-wise Computation），因为矩阵乘法本质上就是乘加运算的累加（Partial Sums）。如果只有矩阵乘法，把数据切块放进 SRAM 里算完再写回 HBM，是再自然不过的高性能计算常规操作。

但 Softmax 打破了这种天然的局部性，成为了分块最大的数学瓶颈。Flash Attention 的最大贡献，就是针对 Softmax 提出了一种数学解决方案。

在标准的 Safe Softmax 计算中，对于输入向量（也就是 Attention 矩阵 $W=QK^T$ 中的一行，一个 Query的总注意力为1）$x \in \mathbb{R}^N$，计算公式如下：
1. **找全局最大值（防溢出）：** $m(x) = \max_{i} x_i$
2. **计算指数并求和：** $l(x) = \sum_{i} e^{x_i - m(x)}$
3. **计算最终概率：** $f(x)_i = \frac{e^{x_i - m(x)}}{l(x)}$

**瓶颈在哪里？**

你要算出一个确切的输出 $f(x)_i$，就**必须**先知道整行的全局最大值 $m(x)$ 和全局分母 $l(x)$。这就意味着：
- 在处理完序列的最后一个 Token 之前，你无法得到真正的 $m(x)$。
- 既然得不到 $m(x)$，你就没法算分母，也没法算最终的概率。
- 因此，在标准 Attention 中，必须把整个巨大的 $S = QK^T$ 矩阵算完，全盘写回 HBM，然后再重新从 HBM 读出来算 Softmax，这就造成了极其高昂的内存读写开销（Memory Bound）。

## 从线段树学习

在这里就假设读者都知道什么是线段树并且有一定的实战经验了。线段树中，父节点可以通过子节点的一些统计信息，来 $O(1)$ 的更新父节点的统计信息。比如说最大值线段树，无论两个子节点对应多长的序列，父节点只需要取两个子节点已经计算好的 max 值，选一个更大的作为自身的 max 值，$O(1)$ 完成序列max的计算。

而 Flash Attention 对 Softmax 的处理也是一样：
1. 记录局部的 $m(x), l(x)$ 
2. 高效合并两个已经计算好的局部的 $m(x), l(x)$ 值

## Online Softmax

从简单的两个合并开始吧，现在我们手头上有两个 tile，分别为 tile1 和 tile2，他们各自的统计参数如下
1. tile1: $m_1(x), l_1(x)$
2. tile2: $m_2(x), l_2(x)$

我们试着只用这四个数，来计算合并 tile1 与 tile2 之后的 tile 的统计参数。
对于 $m(x)$ 很显然：
$$m(x) = max\{m_1(x), m_2(x)\}$$
对于 $l(x)$ 则是比较 tricky，因为是求和所以可以拆分，而指数也可以从 $m(x)$ 替换为 $m_1(x)$ 与 $m_2(x)$ ：
$$
\begin{aligned}
l(x) &= \sum_i e^{x_i - m(x)} \\
 &= \sum_{i \in \text{tile1}}e^{x_i - m(x)} + \sum_{i \in \text{tile2}}e^{x_i - m(x)} \\
 &= \sum_{i \in \text{tile1}}e^{x_i - m_1(x)} e^{m_1(x) - m(x)} + \sum_{i \in \text{tile2}}e^{x_i - m_2(x)}e^{m_2(x) - m(x)} \\
 &= \left( \sum_{i \in \text{tile1}}e^{x_i - m_1(x)} \right) e^{m_1(x) - m(x)} + \left( \sum_{i \in \text{tile2}}e^{x_i - m_2(x)} \right) e^{m_2(x) - m(x)} \\
 &= l_1(x) e^{m_1(x) - m(x)} + l_2(x) e^{m_2(x) - m(x)}
\end{aligned}
$$
至此，我们就有 $O(1)$ 高效合并 tile 的计算方法，可以计算 $m(x), l(x)$ 了。

对于最后的 attention 输出怎么更新？

Flash Attention 在这一步用了一个很巧妙的数学等价，先只存分子，在最后全部的 $l(x)$ 计算完后，再统一进行归一化。

假设 $o_1(x)$ 是之前计算出来的 attention 值，$o(x)$ 是更新后的：

$$o_1(x) = e^{x_i - m_1(x)} V$$
那很显然：
$$o(x) = e^{x_i-m(x)}V =e^{x_i - m_1(x)}\cdot e^{m_1(x) - m(x)}V = o_1(x) e^{m_1(x) - m(x)}$$

## 实际计算流程

1. 针对 Q 进行分块，取若干个（$B_r$个) Q 组成块（外层循环）
2. 针对 K、V 一起进行分块，取若干个 ($B_c$个) 组成块（内层循环）
3. SRAM里存储”内层循环已经走过的tile“的这些信息：
	- 全部的 $o(x)$ ，是个 $(B_r, n B_c)$ 的向量
	- 区域的 $m(x), l(x)$，是两个数
4. 计算一个新的 $B_r \times B_c$ 区域的矩阵乘法，获得“当前tile”的 $o_i(x), m_i(x), l_i(x)$ 
5. 和 SRAM 里的 $o(x), m(x), l(x)$ 进行合并、更新
6. 内层循环结束，将 $O = o(x) / l(x)$ 写回显存

## SRAM 占用计算

1. $B_r$ 个 Q、$B_c$ 个 K 与 $B_c$ 个 V
2. 不断增长的 $o(x)$，最大是 $B_r \times N$ 
3. 两个数 $m(x), l(x)$

相比原版需要直接计算 $N \times N$ 矩阵，这个的实际内存占用降到了 $N \cdot O(N)$ 级别。虽然总量不会少，但它的对 Q 维进行滚动计算（前面乘的 N），对 K、V 维进行分块计算，极大了减少了 SRAM 中的占用。让 SRAM 里的每一个数据都尽可能的是有用的，这就能大大减少显存和 SRAM 中的数据调度开销，很好的缓解了 Memory bound 的 Attention 计算的性能问题。

## 参考资料

![Flash Attention](./flash-attention.png)