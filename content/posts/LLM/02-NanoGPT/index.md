---
title: "「从零开始学大模型」手搓GPT"
subtitle: ""
date: 2026-02-19T23:17:25-08:00
draft: false
toc:
    enable: true
weight: false
categories: ["人工智能"]
tags: ["LLM", "大模型", "Transformer", "笔记"]
---

主要参考：[Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

为了能更加深入理解 Transformer 和 GPT，笔者选择了跟着教程搭一个 GPT 出来。当然只是跟着教程做还是不够，于是就打算写一篇 blog，从自己的角度讲一下到底是怎么个事儿。

## N-gram 语言模型

N-gram 语言模型是一种基于统计的语言模型，核心思想是用前 **n-1** 个词来预测第 n 个词的概率。

**基本原理**

一个句子 $w_1, w_2, \dots, w_m$ 的概率可以用链式法则展开：

$$P(w_1, w_2, \dots, w_m) = \prod_{i=1}^{m} P(w_i \mid w_1, \dots, w_{i-1})$$

但完整的条件历史太长，无法可靠估计，所以 N-gram 做了一个 **马尔可夫假设**：当前词只依赖前 n-1 个词：

$$P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i \mid w_{i-n+1}, \dots, w_{i-1})$$

**常见的 N 值**

- **Unigram (n=1)**：$P(w_i)$，每个词独立，不考虑上下文
- **Bigram (n=2)**：$P(w_i \mid w_{i-1})$，只看前一个词
- **Trigram (n=3)**：$P(w_i \mid w_{i-2}, w_{i-1})$，看前两个词

**概率估计**

最简单的方法是最大似然估计（MLE），直接用语料中的频率计数：

$$P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$$

**历史地位**

N-gram 模型在神经网络语言模型出现之前是 NLP 的主流方法，广泛用于语音识别、机器翻译、拼写纠错等领域。它的局限在于无法捕捉长距离依赖，且随着 n 增大参数量呈指数增长。现代的 Transformer 语言模型本质上解决了同样的"预测下一个词"问题，但用神经网络取代了固定窗口的统计计数。



## BigramModel

在 Andrej Karpathy 的教程里，他从 Bigram 模型开始，一步一步将其改造成了 NanoGPT。于是我这个博客就按照他的顺序，只不过以我自己的理解重新说一遍。

在他的最小 BigramModel 实现中，它仅使用了 `nn.Embedding` 这一个模块。

`nn.Embedding` 本质上是一个查找表（Lookup Table），其内部是一个权重矩阵 $W \in \mathbb{R}^{V \times V}$。当输入的 Token ID 为 $i$ 时，它会返回矩阵的第 $i$ 行，而这一行正好是一个长度为 $V$ 的向量，直接作为下一个 Token 的 logits。

$$W[i, j] = \text{logit}(w_{\text{next}} = j \mid w_{\text{current}} = i)$$

经过 softmax 之后就得到了条件概率 $P(w_j \mid w_i)$。

### 模型实现-Bigram

感觉比较直观，主要是理解一下 `idx` 代表着什么和维度变化：

- 第 0 维：Batch，会有 batch_size 个 (T) 的向量同时传入，并行计算。pytorch 会自动在这一维展开
- 第 1 维：Time step，这就是在输入序列上的位置了，[idx_0, idx_1, idx_2, ...] 
- 具体存的值：是一个 $[0, vocab\_size-1]$ 的整数，代表在字典里的序号
- 输出 `logits`：通过 `nn.Embedding` 查表，获取每一个 `idx` 的**值**所对应的一个 (vocab_size) 的概率向量。
- 所以 `logits` 会额外“长出”一维 (C=vocab_size)，变成 (B, T, C) 的 shape
- 长出来的这一维 C 是对下一位总计 C 个 token 的预测概率，因此是长度是 C。在 `F.cross_entropy` 内部会取出“预测正确的概率”进行交叉熵计算 `loss`

```python
class BigramLanguageModel(nn.Module, Generate):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx (Batch, Time) -> logits (Batch, Time, vocab_size)
        logits = self.token_embedding_table(idx)  
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
```

这一节只放具体模型代码的部分，但已经是完整可运行的“预测模型”了，剩余代码见 [代码实现](#代码实现) 节。

## 从 Bigram 到 N-gram

Bigram 模型存在一个问题，就是它仅仅拿前一个 token 去预测下一个 token，只利用了前一个 token 所提供的信息。

但这显然是完全不够的，我们要把它从一个 token 拓展到多个 token，让多个 token 可以 talk to each other、交换信息，这就是我们这一节要做的事情。

假设现在我们有一个输入序列 $x$，它是一个长度为 $n$ 的序列。我们的目的是要让一个函数使用 $x_0$ 到 $x_i$ 的信息来预测 $x_{i+1}$，那我们要怎么做呢？

我们要做的是一个这样的函数 $f_1(\vec{x})$: 

- 输入 $[x_0, x_1, ..., x_i]$
- 输出 $x_{i+1}$ 的预测值 $\hat{x_{i+1}}$

$$\hat{x_{i+1}} = f_1(x_0, ..., x_i)$$

暂且先不管 $f_1(\vec{x})$ 具体实现，就当它是一个神奇神经网络黑盒，可以预测序列的下一个值。既然是神经网络，那就可以计算 $\text{loss}$ 进行梯度下降了。$\text{loss}$ 也很显而易见，直接对比预测值 $\hat{x_{i+1}}$ 与真实值 $x_{i+1}$ 就行。比如使用交叉熵：

$$ \text{loss} = \text{CrossEntropy}(x_{i+1},\hat{x_{i+1}}) $$

经过梯度下降后，如果 $\text{loss}$ 下降到一个很可观的值，我们就认为这个 $f_1(\vec{x})$ 是个优秀的序列预测函数了。

这就是 N-Gram 模型为什么能做序列预测的原因，也是之后整个 Transformer 模型怎么能一个一个往外吐字的核心。

## 什么是自回归

自回归（autoregressive）是在序列预测里的一个很神奇的特性，我们注意到，对于输入的长度为 $n$ 的训练序列 $[x_0, x_1, ..., x_{n-1}]$，我们实际上可以把他拆成 $n-1$ 组训练数据：

- $[x_0]$ 预测 $\hat{x_1}$
- $[x_0, x_1]$ 预测 $\hat{x_2}$
- $[x_0, x_1, x_2]$ 预测 $\hat{x_3}$
- $[x_0, x_1, x_2, x_3]$ 预测 $\hat{x_4}$
- ......
- $[x_0, x_1, ..., x_{n-3}]$ 预测 $\hat{x_{n-2}}$
- $[x_0, x_1, ..., x_{n-3}, x_{n-2}]$ 预测 $\hat{x_{n-1}}$
- $[x_0, x_1, ..., x_{n-3}, x_{n-2}, x_{n-1}]$ 预测 $\hat{x_n}$

我们设 $y_i = \hat{x_{i+1}}$，则 $\vec{y} = [y_0, y_1, ..., y_{n-1}] = [\hat{x_1}, \hat{x_2}, ..., \hat{x_n}]$

这样我们就再设计 $f_2(\vec{x})$，还是承担着序列预测功能，只不过输出变了：

- 输入 $\vec x = [x_0, x_1, ..., x_{n-1}]$，长度 $n$ 的序列
- 输出 $\vec y = [y_0, y_1, ..., y_{n-1}]$ = $[\hat{x_1}, \hat{x_2}, ..., \hat{x_n}]$，也是长度 $n$ 的序列

由于 $\vec y$ 是函数对 $[x_1, ..., x_n]$ 的预测，且我们的输入对于 $[x_1, ..., x_{n-1}]$ 是已知的，那我们实际上就有了 $n-1$ 个可以用来计算 $\text{loss}$ 的“预测值-真实值”对。

实际计算 $\text{loss}$ 逻辑如下，取有预测值又有真实值的部分进行交叉熵计算：

$$ \text{loss} = \text{CrossEntropy}(\mathbf{y}[:-1], \mathbf{x}[1:]) $$

多说一点，在这里 $f_1$ 和 $f_2$ 的底层逻辑是完全相同的，$f_2$ 可以视作是 $n$ 个 $f_1$ 并行的在进行序列预测。而这一点在 Tensor 处理中可以很好的被“广播”，$f_1$ 会自动沿着多出来的维度进行展开、并行计算。

## 从 f(x) 到矩阵

我们先假定一个简单情况：让 $f_2$ 预测 $x_{i+1}$ 会等于 $x_0$ 到 $x_i$ 的求和。

$$ f_2(\vec{x}) = \sum_{j = 0}^{i} x_j $$

可以用简单的循环来实现 $f_2(\vec{x})$：

```python
def f2(x: list[float]) -> list[float]:
    y = [0 for _ in range(len(x))]  # 构造长度为 i 的 [0, 0, 0, 0]
    for j in range(len(x)):     # 对 y_i
        for i in range(j + 1):  # 遍历 x_0 ~ x_i
            y[j] += x[i]        # 求和
    return y
```

当然有扎实线性代数基础的你会注意到，这和矩阵乘法很像，我们只需要构造这样的一个矩阵 $\text{W}_{f_2}$ （假设 $n=8$）：

$$
\text{W}_{f_2} = \begin{bmatrix}
1&0&0&0&0&0&0&0\\
1&1&0&0&0&0&0&0\\
1&1&1&0&0&0&0&0\\
1&1&1&1&0&0&0&0\\
1&1&1&1&1&0&0&0\\
1&1&1&1&1&1&0&0\\
1&1&1&1&1&1&1&0\\
1&1&1&1&1&1&1&1\\
\end{bmatrix},

\vec{x} = \begin{bmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \\ x_6 \\ x_7
\end{bmatrix},

\vec{y} = \begin{bmatrix}
y_0 \\ y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5 \\ y_6 \\ y_7
\end{bmatrix}
$$

就有：

$$ \text{W}_{f_2} \vec{x} = \vec{y} $$

> **Causal Mask**：
> 
> 使用对 $x_0$ 到 $x_i$ 求和，本质上是为了防止 $y_i$ 去偷看未来的 $x_{i+1}$，$y_i$ 只能受到 $x_0$ 到 $x_i$ 的“因果”的影响。这个下三角矩阵被称作 Causal Mask（因果遮罩？），后面会有它的变体，但本质上还是在训练时约束因果关系。

我们再往前一步。因为求和会让 $x_i$ 的数值不断增大，我们把求和改成取平均：

$$ f_2(\vec{x}) = \frac{1}{i+1} \sum_{j = 0}^{i} x_j $$

那么这个矩阵就会变成如下的一个下三角矩阵$\text{W}_{f_3}$ （假设 $n=8$）：

$$
\text{W}_{f_3} = \begin{bmatrix}
1/1&0&0&0&0&0&0&0\\
1/2&1/2&0&0&0&0&0&0\\
1/3&1/3&1/3&0&0&0&0&0\\
1/4&1/4&1/4&1/4&0&0&0&0\\
1/5&1/5&1/5&1/5&1/5&0&0&0\\
1/6&1/6&1/6&1/6&1/6&1/6&0&0\\
1/7&1/7&1/7&1/7&1/7&1/7&1/7&0\\
1/8&1/8&1/8&1/8&1/8&1/8&1/8&1/8\\
\end{bmatrix}
$$

这个矩阵，就是我们的 Attention 矩阵的雏形了，它代表着 $token_i$ 的注意力平均分散到 $token_0, token_1, ..., token_{i-1}$ 身上。

## 加入 Attention Matrix

在前面，我们已经构造了最基础的 BigramModel，并讲解清楚了 Attention Matrix 这个矩阵所代表的数学含义。

在这一节，我们将两者结合一下：
1. 我们还是采用一个 `nn.Embedding` 对输入进行处理，从 `token_idx` 变为“意义”空间的向量 `emb_idx`
2. 因为不止 `token_idx` 本身有含义，它在句子中的位置也代表一定的意义。所以我们会增加一个 `nn.Embedding` 对输入进行位置编码，从 `token_pos` 变为 `emb_pos`
3. 将两个“意义”简单相加，代表“这个位置出现这个token的意义” `emb_vec`
3. 接着使用 Attention Matrix 矩阵乘法，让每个“意义”相互交流
4. 最后需要从“意义”空间解码，变回 `token_idx` 进行输出

> **意义空间** 与 $d_{model}$：
>
> 可以理解为在模型的内部，它使用一个 $d_{model}$ 维的向量对 token 进行表示、理解。同时我们的 Attention Matrix 也是一个 ($d_{model}$, $d_{model}$) 的矩阵，它让这些不同的“意义”可以进行相互交流。在英文交流中 $d_{model}$ 张成的空间被称作 “Embedding Space”，*“嵌入空间”*，向量则是被称作“嵌入向量”，但我个人喜欢把他直接理解为“意义”，很直观。
> 
> 当然我们还注意到这直接使用了 model 作为下标，它表述的是 dimention of this model。笔者认为这是这一个模型的最核心参数，才能有此殊荣。这一参数或许直接决定了模型的能力。

## Head 的概念

在这里我们还没有加上 Attention 中的 QKV 矩阵，但是 Head 的 雏形已经出现了。在这里可以简单理解为：我们从某一个视角去看这个序列，看到里面所蕴含的信息。

我们在 v1 版本中所做的事，就是让这个 Head 直接去学习：”你能不能从序列中看出来下一个 Token 的信息“。每一个 Head 都包含了独立的 Attention Matrix，用于让 head 从输入中”察觉“到信息。

### 模型实现-NanoGPTv1

```python
d_model = 32
class NanoGPTv1(nn.Module, Generate):
    def __init__(self, vocab_size):
        super().__init__()
        # token_idx -> emb_idx
        self.tok_emb_table = nn.Embedding(vocab_size, d_model)
        # position -> emb_pos
        self.pos_emb_table = nn.Embedding(block_size, d_model)

        # Language Model Head
        # 可以理解为 Un-embedding，将 emb_vec 转换回 token_idx
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.tok_emb_table(idx)  # (B, T, d_model)
        pos_emb = self.pos_emb_table(torch.arange(T, device=device))  # (T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)
        # 取前面所有的 emb_vec 的平均
        W_f3 = torch.tril(torch.ones((T, T), device=device))  # (T, T) 下三角矩阵
        W_f3 = W_f3 / torch.sum(W_f3, dim=1, keepdim=True)
        x = W_f3 @ x # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
```


## 加入 Softmax

$$ \text{Softmax}(y_i) = \frac{ e^{y_i} }{ \sum_j e^{y_j} } $$

这里就不过多赘述为什么要加 Softmax 了，简单来说就是为了我们后面从特殊（取平均）拓展到一般（任意 Attention值）但要防止 Attention 数值爆炸做的归一化。

在这里先实现一个 $\text{W}_{f_4}$，使得 $\text{Softmax}(\text{W}_{f_4}) = \text{W}_{f_3} $。显然，我们需要的是一个这样的 $\text{W}_{f_4}$：

$$
\text{W}_{f_4} = \begin{bmatrix}
1&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty\\
1&1&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty\\
1&1&1&-\infty&-\infty&-\infty&-\infty&-\infty\\
1&1&1&1&-\infty&-\infty&-\infty&-\infty\\
1&1&1&1&1&-\infty&-\infty&-\infty\\
1&1&1&1&1&1&-\infty&-\infty\\
1&1&1&1&1&1&1&-\infty\\
1&1&1&1&1&1&1&1\\
\end{bmatrix}
$$

对 $\text{W}_{f_4}$ 使用 $\text{Softmax}$，是将 $\text{W}_{f_4}$ 的每一行都拿去做 $\text{Softmax}$，会得到：


$$\text{Softmax}\left(
\begin{bmatrix}
1&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty\\
1&1&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty\\
1&1&1&-\infty&-\infty&-\infty&-\infty&-\infty\\
1&1&1&1&-\infty&-\infty&-\infty&-\infty\\
1&1&1&1&1&-\infty&-\infty&-\infty\\
1&1&1&1&1&1&-\infty&-\infty\\
1&1&1&1&1&1&1&-\infty\\
1&1&1&1&1&1&1&1\\
\end{bmatrix}
\right) = 
\begin{bmatrix}
1/1&0&0&0&0&0&0&0\\
1/2&1/2&0&0&0&0&0&0\\
1/3&1/3&1/3&0&0&0&0&0\\
1/4&1/4&1/4&1/4&0&0&0&0\\
1/5&1/5&1/5&1/5&1/5&0&0&0\\
1/6&1/6&1/6&1/6&1/6&1/6&0&0\\
1/7&1/7&1/7&1/7&1/7&1/7&1/7&0\\
1/8&1/8&1/8&1/8&1/8&1/8&1/8&1/8\\
\end{bmatrix}
$$

对 `NanoGPTv1` 代码做少量更改即可：

```python
# 构造 W_f4：下三角为 0，上三角为 -inf
tril = torch.tril(torch.ones((T, T), device=device))
W_f4 = torch.zeros((T, T), device=device)
W_f4 = W_f4.masked_fill(tril == 0, float('-inf'))
# Softmax(W_f4) = W_f3
W_f3 = F.softmax(W_f4, dim=1)
x = W_f3 @ x  # (B, T, d_model)
```

## 青春版 Attention

到这一步，我们就可以让模型去学习这个 W 矩阵了。我们把它从固定的平均矩阵，替换为一个可学习的 $\text{W}_{f_5}$，然后交给模型去学习。

### 模型实现-NanoGPTv2

```python
class NanoGPTv2(nn.Module, Generate):
    def __init__(self, vocab_size):
        super().__init__()
        # token_idx -> emb_idx
        self.tok_emb_table = nn.Embedding(vocab_size, d_model)
        # position -> emb_pos
        self.pos_emb_table = nn.Embedding(block_size, d_model)

        self.W_f5 = nn.Parameter(torch.tril(torch.ones((block_size, block_size), device=device)))  # (T, T) 下三角矩阵
        # Language Model Head
        # 可以理解为 Un-embedding，将 emb_vec 转换回 token_idx
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.tok_emb_table(idx)  # (B, T, d_model)
        pos_emb = self.pos_emb_table(torch.arange(T, device=device))  # (T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)
        tril = torch.tril(torch.ones((T, T), device=device))
        wei = self.W_f5[:T, :T].masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=1)
        x = wei @ x  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
```

## QKV

在我们的 `NanoGPTv2` 中，存在一个局限性。我们学习得到的 $\text{W}_{f_5}$ 这个 Attention Matrix 是一个固定的矩阵————不管输入什么内容，同一个位置的权重都是一样的。位置5永远以固定的权重去参考位置1、2、3、4，和位置5上是什么词，也不管1、2、3、4上实际是什么词。

为了实现能“根据具体内容调整 Attention Matrix”，论文把 W 矩阵拆分为 Q、K 两部分：

- **Query**: 我在寻找什么（根据“我是什么”而变化）
- **Key**: 你能提供什么（根据“你说什么”而变化）
- 当我 ($x_i$) 要找的东西 ($q_i$) 和你($x_j$)能提供的东西 ($q_j$) 匹配度更高时，我就更关注你($W_{ij}$ 值更大)
- 很显然的，$ W_{ij} = \vec{q}_i \cdot \vec{k}_i $ 可以计算 $\vec q_i$ 与 $\vec k_i$ 之间的相似度（匹配度）

在实现方面，从 $x_i \to q_i, \ x_i \to q_i$ ，模型在 (B, T) 两个维度上并行:

- 将“意义空间”的所有 batch、所有位置的 $x_i$ 组成的 $X$: (B, T, $d_{model}$)
- $X$ 右乘 $W_Q$, $W_K$: ($d_{model}$, $d_{k}$)
- 得到 $Q$, $K$: (B, T, $d_{k}$)
- $ W = Q K^T $: (B, T, T)

注意此处的 $X$ 是右乘 $W_Q$、$W_K$，这和前面直接用 weight 对 $X$ 进行变换（左乘）是不一样的。因为这里实际上是对 $X$ 做线性投影，而不是加权聚合

$$ Q = X W_Q \\ K = X W_K $$

> **查询空间** 与 $d_k$:
>
> $d_k$ 代表着 $d_{key}$，是注意力头用来查询、匹配的 qeury 与 key 的向量维度。可以认为 Q 和 K 是最终注意力矩阵 $W$ 的一个 $d_{k}$ 的低秩分解。虽然看起来 $d_{k}$ 不会影响 $W$ 的形状 (T, T) ，但实际上会影响注意力矩阵 $W$ 的秩（可以理解为有效信息含量）
>
> 在实际操作中，$T >>d_{k}$。比如说 GPT-2 的 $d_{k} = 64$，$T = 1024$，所以 $QK^{\top}$ 一直都会是矩阵的低秩分解。
>
> 因此，单个注意力头只能表达秩为 $d_{k}$ 的注意力模式。我们就需要通过多注意力头来捕获不同的注意力模式，结合起来后，表达能力会更强。

**Value**: 在 Query 和 Key 之外，论文作者还增加了 Value 这一中间量

- 它所表示的是：我给你哪些信息
- 也是对嵌入空间中的 $x_i$ 进行一个线性变换，变为**值空间**中的 $v_i$ 向量
- $W_V$: ($d_{model}$, $d_v$)
- $d_v$ 可以不等于 $d_k$，但是论文中 $d_v = d_k = d_{model} / \text{num\_heads}$

$$ V = X W_V $$

到这里，我们就实现了最关键的部分：Attention

$$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$

> 为什么除以 $\sqrt{d_k}$ ?
>
> 论文原话的意思是：当 $d_k$ 比较大时，Q和K的点积结果会变得很大，把softmax推到梯度极小的饱和区，导致训练不动。
>
> $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$，假设 $q_i, k_i$ 是均值0，方差1的随机变量，则 $q_i k_i$ 的方差是 1，求和后方差 $d_k$，标准差 $\sqrt{d_k}$ 
>
> 除以 $\sqrt{d_k}$ 把方差拉回到 1，让softmax的输出分布更平滑，梯度正常流动。本质就是个数值稳定性的trick。

### 代码实现 NanoGPTv3

在 pytorch 实现中，$W_Q, W_K, W_V$ 分别使用一个全连接层 `nn.Linear` 来表示。

```python
d_model = 384
d_k = 64
d_v = 64
class NanoGPTv3(nn.Module, Generate):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb_table = nn.Embedding(vocab_size, d_model)
        self.pos_emb_table = nn.Embedding(block_size, d_model)
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.lm_head = nn.Linear(d_v, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.tok_emb_table(idx)  # (B, T, d_model)
        pos_emb = self.pos_emb_table(torch.arange(T, device=device))  # (T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)
        
        Q = self.W_Q(x)  # (B, T, d_k)
        K = self.W_K(x)  # (B, T, d_k)
        V = self.W_V(x)  # (B, T, d_v)
        W = Q @ K.transpose(-2, -1) * (d_k ** -0.5)
        
        tril = torch.tril(torch.ones((T, T), device=device))
        W = W.masked_fill(tril == 0, float('-inf'))
        W = F.softmax(W, dim=-1)

        y = W @ V  # (B, T, d_v)
        logits = self.lm_head(y)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
```

## 目标概览

<center><img src="./gpt2.png" width="80%" /></center>

从这之后，我们将参照这个图，一点一点实现整个完整的 Transformer 架构。

## Multi-Head Attention

为了弥补之前提到的 $d_k$ 限制了注意力模式的问题，Transformer架构加入了多头注意力：

- $h$ (`num_head`) 个结构相同、权重不同的注意力头
- 同时进行，输出“各个注意力头获得的信息” (B, T, d_v)
- 把每个头的输出 concat 到一起 (B, T, d_v * h == d_model)
- 并经过全连接混合，最后再从嵌入空间解码到 vocab_idx

### 代码实现 Multi-Head

在这之后我将只更新模块，将其拼接到模型中。在这一节我将 Head 抽象成一个 `class` ，方便复用，并简单添加了 MultiHead 的代码：

```python
d_model = 384
d_k = 64
d_v = 64

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_Q(x)  # (B, T, d_k)
        K = self.W_K(x)  # (B, T, d_k)
        V = self.W_V(x)  # (B, T, d_v)
        W = Q @ K.transpose(-2, -1) * (d_k ** -0.5)
        
        tril = torch.tril(torch.ones((T, T), device=device))
        W = W.masked_fill(tril == 0, float('-inf'))
        W = F.softmax(W, dim=-1)

        y = W @ V  # (B, T, d_v)
        return y

class MultiHead(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]  # List of (B, T, d_v) * num_heads
        concatenated = torch.cat(head_outputs, dim=-1)  # (B, T, num_heads * d_v) == (B, T, d_model)
        output = self.proj(concatenated)  # (B, T, d_model)
        return output
```

## Feed-Forward Network

在 Multi-Head Attention 之后，所有的嵌入空间的向量会经过一个叫作 Feed-Forward Network 的神经网络。

在 GPT-2 中，这个环节叫做 Multi-layer Projection (MLP)，不过本质上是一个东西。

它的构成比较简单：
- 一个升维的矩阵 $ W_{\uparrow},B_{\uparrow} $ （带bias）
- 一个激活函数 `nn.GeLU()`
- 一个降维的矩阵 $ W_\downarrow,B_\downarrow $

向量不再互相交流，而是并行进行同一处理。根据 3Blue1Brown 的说法，可以把这个环节理解为：模型对每一个嵌入空间的向量提出多个不同的问题，然后根据这些问题的回答来更新向量。问题矩阵 $W_{\uparrow}$ 矩阵列数($ d_{ff} =  4 \times d_{model} $) 可以认为是模型对每个向量的问题数量。

$$\overrightarrow{E} \to W_{\uparrow},B_{\uparrow} \to \text{GeLU} \to W_\downarrow,B_\downarrow \to \overrightarrow{E}’$$

最后，还有一个残差结构，$\overrightarrow{E}_{out}=\overrightarrow{E}+\overrightarrow{E}'$

### 代码实现 FFN

```python
d_ff = 4 * d_model
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)
```

## Block

有了 Multi-Head Attention 和 Feed-Forward Network，要实现 Transformer 中的一个 Block，我们还需要 LayerNorm 和 Dropout。

LayerNorm 的目的主要是为了稳定训练，它将每个 token 的特征归一化。

Dropout 的目的是随机丢弃掉一部分的值，防止网络过拟合。

在实现中还有一个残差连接。它是为了在前期训练的时候，让梯度有一条“高速公路”直接流回到比较前面的网络，方便层数较多的网络进行训练。

```python
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.heads = MultiHead(num_heads = d_model // d_v)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.heads(self.ln1(x))) + x  # (B, T, d_model)
        x = self.dropout(self.ffn(self.ln2(x))) + x     # (B, T, d_model)
        return x

class NanoGPT(nn.Module):
    def __init__(self):
        # ...
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_blocks)])
        # ...
```

## MultiHead优化

由于 $d_k = d_v$，我们注意到，$W_Q, W_K, W_V$ 都是 ($d_{model}$, $d_k$) 我们在数学上可以把他们拼接成一个 ($d_{model}$, $3 \times d_k$) 的大矩阵，同时计算。这样可以减少 GPU 的 IO 时间，加速训练与推理速度。

同时，我们还注意到，不仅 QKV 可以拼在一起计算，每个 head 实际上也是在算同样的东西，可以全部拼成一个矩阵进行计算 ($d_{model}$, $3 \times d_k \times h$) = ($d_{model}$, $d_{model}$) 的巨大矩阵。

算完 QKV 的大结果矩阵再变换一下，提取出一个 h 维，让最后两维正好分别是每个 Head 的 Q、K、V 的两维。这样还是可以直接对所有的 QKV 同时进行矩阵乘法。

类似地，FFN 环节也可以这样并行计算。

```python
class Attention(nn.Module):
    def __init__(self, config: Hyperparameters):
        super().__init__()
        # query, key, value projections for all heads, but in a batch
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # output projection
        self.proj = nn.Linear(config.d_model, config.d_model)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.d_h = config.d_model // config.d_k
        self.d_model = config.d_model
        self.dropout = config.dropout
        # TODO: flash attention

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(
                1, 1, config.block_size, config.block_size
            )
        )
    
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)

        q, k, v = self.qkv(x).split(self.d_model, dim=2)  # (B, T, d_model) x 3
        # split into heads
        q = q.view(B, T, self.d_h, C // self.d_h).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.d_h, C // self.d_h).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.d_h, C // self.d_h).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (C // self.d_h) ** -0.5  # (B, nh, T, T)
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))  # (B, nh, T, T)
        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.proj(y)  # (B, T, C)
        y = self.resid_dropout(y)
        return y
```

## torch.compile & TensorCore

两句话，再让模型加速50%

```python
torch.set_float32_matmul_precision('high')
model = torch.compile(model)
```

## 代码实现

```python
# 在所有代码之前
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 数据加载

```python
# 数据集获取、处理
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))  # 以字符作为 token 
vocab_size = len(chars)          # 根据字符获取字典大小


# 构建字符到索引的映射，encoder、decoder
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]           # 字符    -> 字典索引
decode = lambda l: ''.join([itos[i] for i in l])  # 字典索引 -> 字符


# 区分训练集和测试集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# 辅助函数
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

### 模型

具体代码散落在文章的各处，取来，用之。

[BigramModel](#模型实现-bigram),
[NanoGPTv1](#模型实现-nanogptv1),
[NanoGPTv2](#模型实现-nanogptv2),
[NanoGPTv3](#代码实现-nanogptv3),


### 训练

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = NanoGPTv1(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(train_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### 推理

```python
class Generate():
    @torch.no_grad()  # 生成文本时不需要计算梯度，会节省内存和计算资源
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # 获取输入的最后 block_size 个字符
            idx_cond = idx[:, -block_size:]
            # 前向传播调用 forward()，获取下一个字符的概率分布
            logits, _ = self(idx_cond)
            # 获取最后一个时间步的输出
            logits = logits[:, -1, :]
            # 将 logits 转换为概率分布
            probs = F.softmax(logits / temperature, dim=-1)
            # 从概率分布中采样下一个字符的索引
            next_idx = torch.multinomial(probs, num_samples=1)
            # 将新生成的字符索引添加到输入序列中
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
```

### 代码存档

右键另存为

[各版本 NanoGPT](./blog.ipynb)

[train](./train.py), [model](./model.py)