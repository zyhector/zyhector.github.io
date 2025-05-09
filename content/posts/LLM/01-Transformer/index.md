---
title: "「从零开始学大模型」Transformer"
subtitle: ""
date: 2025-05-09T17:46:21+08:00
draft: false
toc:
    enable: true
weight: false
categories: ["人工智能"]
tags: ["LLM", "大模型", "Transformer", "笔记"]
---


## Transformer是做什么的

在GPT等模型中，Transformer模型输入文字、音频、图像等数据，并对文本中下一词出现的概率做预测。选择概率最高的词，追加到输入文本的后面，再将补全后的文本重新作为输入，如此往复，实现文本的补全。

## 处理流程

模型从输入到输出的处理流程如下：

* Embedding
* 注意力模块
* 前馈层
* （注意力模块+前馈层）x N
* Unembedding
* Softmax
* 输出

<center><img src="./gpt2.png" width="80%" /></center>

## Embedding

输入被分割成小段小段，这些小段称之为Token。输入文字则分割成一个个单词、标点或词根，图片则是小块图片、声音切分为小段音频。

Embedding模块为一个矩阵，包含着 所有可能的Token（字符串）$\to$高维向量$\overrightarrow{E}$。GPT3有12,288（注意到3x4096=12,288）嵌入个维度，经过训练，Token被embed成的向量$\overrightarrow{E}$有具体的语义意义，在嵌入空间中，某些特定的方向是有意义的，比如$\overrightarrow{man}-\overrightarrow{women} \approx \overrightarrow{king}-\overrightarrow{queen}$，在作差得到的这个向量的方向上，可能就表示性别，或更加男性化或更加女性化。

GPT3有50,257个支持的Token，因此Embedding矩阵$W_E$有12,288x50,257个参数。

## Attention

一个单词的真实意义可能会被其上下文影响，甚至是很远的上下文。因此使用Attention模块进行处理，使每一个$\overrightarrow{E}$有更精确的含义。这一环节总的能处理的$\overrightarrow{E}$数量有限制，被称之为上下文长度，GPT3的上下文长度为2,048，因此数据有2,048列，每个有12,288维。

比如说，“Tower”这个词，只能被Embedding矩阵粗略地解码为“塔”，或者“高大（Towering）”，但如果输入的文字为“Eiffel Tower”，该单词就明确的指向埃菲尔铁塔，“Eiffel”这个词更新了“Tower”单词的意思，对应地，更新了该Token对应的$\overrightarrow{E}$为$\overrightarrow{E}'$。进一步地，如果输入为“Miniature Eiffel Tower”，就该更新为埃菲尔铁塔模型，和“高大”就没什么关系了。

Attention模块不只是“更新单词意思”，更重要的是允许了$\overrightarrow{E}$将其自身的含义“注入”到别的$\overrightarrow{E}$中去，丰富其他$\overrightarrow{E}$代表的内容。这一机制就使得注意力模块可以让前文的所有$\overrightarrow{E}$的信息，注入到最后一个的$\overrightarrow{E}$中去，综合前文全部的信息以产生新的Token，使得“Generative”这一特性变为可能。

## 注意力机制

每个$\overrightarrow{E}$会计算一个Query向量$\overrightarrow{Q}$，只有128维。

要计算这个向量，需要矩阵$W_Q$，乘以$\overrightarrow{E}$，得到查询空间中的向量$\overrightarrow{Q}$。

同时，还有一个矩阵叫做键矩阵（Key Matrix）$W_K$，与$\overrightarrow{E}$向量相乘，得到“keys”向量，$\overrightarrow{K}$。可以理解为对Query的回答（？）。当Key与Query的方向对其时，就能认为他们相匹配。

对一串$\overrightarrow{E}$产生的所有的$\overrightarrow{K}$与$\overrightarrow{Q}$做点积，就可以知道那些Key与Query匹配，称之为Keys“注意到了”Query。

点击产生的矩阵成为注意力模式（Attention Pattern）。要使用这个矩阵，先对每个Query对应的一列，使用Softmax进行归一化。之后，这一列中每个数所代表的，是Query和Keys的相关性，可以进行加权求和了。

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

($d_k$为空间维度，为了维护点积数值的稳定性，将每个结果除以空间维度的平方根)

为了提高训练速度，有个很tricky的小技巧。在训练模型中，模型同时可以利用多个Token进行训练。比如，将长度为50的Token序列$\{t_0,t_1,...,t_{49}\}$作为训练集送入模型，模型可以同时训练利用$\{t_0\}$预测$t_1$，利用$\{t_0,t_1\}$预测$t_2$，利用$\{t_0,t_1,t_2\}$预测$t_3$，一直到利用$\{t_0,t_1,...,t_{48}\}$预测$t_{49}$。

但如果执行使用$\{t_0,t_1,...,t_{48}\}$预测$t_{49}$的任务时，模型直接向$t_{49}$进行询问，并将这个权重设置为最大，不就可以轻松获取到该输入什么可以取得最大的分数，就漏题了。为此，需要将注意力模式这一矩阵变为上三角矩阵，将$M_{i,j(i>j)}$全部置零。当然这一操作会破坏和为1的特性（不能加权求和）。因此将数据在通过Softmax之前，全部置为负无穷，这样在Softmax之后，得到一个上三角、且归一化的矩阵。这一种处理叫做“Masking”掩码。

这个矩阵的大小为上下文长度的平方。因此这个矩阵的大小就成为上下文长度的限制的主要因素。

要实现更新$\overrightarrow{E}$，要知道Key向量$\overrightarrow{E}_K$该怎么更新Query向量$\overrightarrow{E}_Q$，具体的说，该对Query向量的值做何种改变。这里就需要Value矩阵$W_V$，乘以Key向量$\overrightarrow{E}_K$，得到值向量$\overrightarrow{V}$，并加到$\overrightarrow{E}_Q$向量中。

$$\overrightarrow{E}_{Q_i}' = \overrightarrow{E}_{Q_i} + \sum_{j \in context} \mathrm{Attention}(Q_i,K_j,V_j) $$

$W_Q,W_K$矩阵分别为12,288（嵌入空间维度）x128（查询空间维度）=1,572,864个参数。

$W_V$矩阵理论上为12,288x12,288哥参数，但这样有点太大了。实际上，$W_V$矩阵的参数量为$W_Q,W_K$参数量之和，通过将$W_V$低秩分解为两个小矩阵之积，如$W_{1(12,288\times 128)}W_{2(128\times 12,288)}=W_{V(12,288\times 12,288)}$。3b1b将这里的$W_1$称之为$\mathrm{Value}\uparrow$矩阵，$W_2$称之为$\mathrm{Value}\downarrow$矩阵（这就是LoRA的原理）。

因此，$W_K,W_Q,W_{V_\uparrow},W_{V_\downarrow}$，每个有1,572,864个参数。一个注意力头有6,291,456个参数。

## 多头注意力机制

而多头注意力机制就是有并行的多个注意力头（可以理解为不同的通道），每个注意力头都有自己的$W_K,W_Q,W_V$。每个注意力头$j$都会对某个token $t_i$给出一个向量变化量$\Delta \overrightarrow{E}_i^{(j)}$，所有的$\Delta \overrightarrow{E}_i^{(j)}$都要加到$\overrightarrow{E}$上。

GPT3有96个注意力头，合计603,979,776个参数。

在论文中，所有的$W_{V_\uparrow}$会拼在一起，合称为“输出矩阵”，与多头注意力模块关联，而$W_{V_\downarrow}$则是分配到每一个注意力头的输出中，对输出进行降维。

GPT3还有96层，因此还要将参数总数乘以96，得到57,982,058,496。但这些参数实际上只占总参数的1/3，参数的大头还在MLP模块中。

## 多层感知/前馈层

Multilayer Perceptron, MLP

向量不再互相交流，而是并行经过同一处理。有点像对每个向量提出多个不同的问题，然后根据这些问题的答案来更新向量。这一环节是大模型真正“记忆”知识的地方。

问题矩阵$W_{\uparrow}$矩阵行数可以认为对每个向量的问题数量，在GPT3中，行数为49,152=(4x12,288)嵌入空间的四倍。$W_{\uparrow}$将$\overrightarrow{E}$升维到问题空间。

使用ReLU进行激活。

还有$W_{\downarrow}$，重新将维度降回嵌入空间。

$$\overrightarrow{E} \to W_{\uparrow},B_{\uparrow} \to ReLU \to W_\downarrow,B_\downarrow \to \overrightarrow{E}'$$

最后，还有一个残差结构，$\overrightarrow{E}_{out}=\overrightarrow{E}+\overrightarrow{E}'$

这一环节为一个多层的全连接网络，网络上没有什么特别的。

结束MLP层后，还有一层Layer Norm层。

$W_{\uparrow}$参数为4x12,228x12,228=603,979,776。$W_{\downarrow}$参数量一致，只不过行列维度互换。合计1,207,959,552个参数。再乘以96层，为115,964,116,116,992个参数。

与注意力的参数加起来，得到GPT的175,181,291,520（175B）参数


## Unembedding

将最后一个向量的值，解码为各个token的概率，再经过softmax（tempure在这一阶段注入，低的tempure会导致Softmax将“赢者通吃”，全部倾斜到大的输入值上），取最可能的token。

## 为什么能存那么多知识

MLP层中，问题空间只有约5万维。如果信息全部以正交的方式存储，只能存5万个知识，但显然GPT比这聪明多了。实际上，MLP中可以将角度为89°~90°的向量，当做几乎正交，进而以这种方式来存储知识。根据“约翰逊-林登斯特劳斯引理”，能在空间中塞进“几乎”垂直的向量数量，随着空间维度的增长指数型上升。这也是为什么能在5万维中存储这么多知识。而通过增加维度（Scaling），知识量会急剧增加，模型也就会更加聪明。

## 参考

论3b1b为什么是神！

深度学习第5章 [BV13z421U7cs](https://www.bilibili.com/video/BV13z421U7cs)

深度学习第6章 [BV1TZ421j7Ke](https://www.bilibili.com/video/BV1TZ421j7Ke)

深度学习第7章 [BV1aTxMehEjK](https://www.bilibili.com/video/BV1aTxMehEjK)