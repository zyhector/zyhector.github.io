---
title: "「从零开始学大模型」TRL GRPOTrainer 源码导读"
subtitle: ""
date: 2026-04-16T12:13:00-07:00
draft: false
toc:
    enable: true
weight: false
categories: ["单卡道场"]
tags: ["LLM", "RL", "GRPO", "读源码"]
---

这是在 Hugging Face 的 TRL [Quickstart](https://huggingface.co/docs/trl/quickstart) 界面里，GRPO 训练的示例。今天从这里开始，探索一下 HF 的 GRPOTrainer 是怎样实现的，涵盖普通 Training loop 与 RL/GRPO 中的 Rollout、Training 环节，为之后我们手搓 GRPO Training Loop 打下基础！

```python
from trl import GRPOTrainer
from datasets import load_dataset
from trl.rewards import accuracy_reward

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=load_dataset("trl-lib/DeepMath-103K", split="train"),
    reward_funcs=accuracy_reward,
)
trainer.train()
```

## 继承链与分工

`GRPOTrainer` 继承自 `_BaseTrainer`，`_BaseTrainer` 继承自 `transformers.Trainer`。

```text
transformers.Trainer -> _BaseTrainer -> GRPOTrainer
```

在这条链上，主要的 training loop 由基类 `Trainer` 实现，而 `GRPOTrainer` 则是重写了几个关键函数，让 GRPO 的 rollout 生成、group 打分与 loss 计算得以实现。

后文我们详细拆解一下这三个类的功能，定位到 `trainer.train()` 这一个函数的具体执行逻辑是什么，调用了哪些其他函数，详细理解一下 RL 的 training loop 在代码上是怎么实现的。
## `Trainer` 基类
### 功能模块分工
`class Trainer` 定义于 transformers 的库内，整个 `trainer.py` 有整整 4412 行，相当复杂。根据代码里的分段注释，`class Trainer` 的代码自上而下，实现了这些功能：

1. Initialization & Validation - 初始化与参数检验
2. Data Loading - 加载数据
3. Optimizer & Scheduler & Learning rate - 优化、调度、学习率
4. Training - 训练模块
5. Training Utilites - 训练工具
6. Evaluation & Prediction - 验证与推理
7. Checkpoint Saving - 训练checkpoint保存
8. Checkpoint Resuming - 训练checkpoint加载、恢复
9. Saving & Serialization - 整个模型的保存与序列化
10. Logging & Metrics - 日志与性能测试
11. Hub Integration - Hugging Face 集成
12. Hyperparameter Search - 自动尝试超参数组合
13. Callbacks - 添加删除回调函数
14. Utilities - 小工具

就算只是列举功能模块也足足有 14 项，更别说每个模块里少则二三多则十几个的函数了。好在我们目标是学习 training loop，只需要看 Training 模块就好。
### Training Loop

Training 模块里面的函数也不少，有这些：


1. `train()` - 训练入口
2. `_inner_training_loop()` - 实际训练循环发生的地方，逐 epoch 的 for 循环
3. `_init_training_state()` - 训练状态初始化
4. `_run_epoch()` - 每个 epoch 做什么，两层循环：外更新 / 内 micro-batch 拆分
5. `_finalize_training()` - 收集 metric 数据、清理、生成 output 等等
6. `training_step()` - 在 run_epoch() 的内层循环调用，forward + backword
7. `compute_loss()` - 每一个 training_step() 调用一次，forward 与计算 loss
8. `compute_loss_context_manager()` - 计算 loss 的辅助函数
9. `autocast_smart_context_manager()` - autocast 的辅助函数
10. `_maybe_log_save_evaluate()` - log 与记录


简化一下逻辑，可以这么理解：
```python
# 外部 call trainer.train()
# call _inner_training_loop()
for epoch in range(epoches): # epoch 循环
	# call _run_epoch()
	for update_step in range (update_steps):  # 外层：权重更新循环
		for i in range(batch_samples): # 内层：micro batch
			# call training_step()
			inputs = self._prepare_inputs()
			loss = self.compute_loss(inputs)
			self.accelerator.backward(loss)
		# 每个 update_step, 或整个 epoch 的最后一步
		optimizer.step()    # 权重更新
		lr_scheduler.step() # 学习率更新
		model.zero_grad()   # 梯度清零
```

简单的三层循环：epoch、update_step、batch。这就是核心的训练循环了，很朴素，除了把大 batch 拆成显存装得下的 micro batch，和其他常见的训练循环没有什么特别的。

在子类 GRPOTrainer 中，一样用的是这个循环，仅仅是重写了其中的几个函数。我们理解了基类的训练循环，就可以详细深入 GRPOTrainer，看看为什么重写那几个函数就能把普通的训练变成 GRPO 了。
## `_BaseTrainer` 类

虽然这个模块基本上什么都没做，但既然在继承链上还是顺带提一嘴。它继承自 `Trainer` ，仅仅是重写了在基类的 Hub Integration 这一模块 的 `create_model_card()` 函数。

父类 HF Trainer 的关注点是通用 ML 模型发布：license、language、tasks、dataset 这些是 HF Hub 检索和展示需要的元数据。

TRL _BaseTrainer 的关注点是论文算法复现：TRL 里每个 trainer 对应一篇论文（PPO、DPO、GRPO…），所以卡片里要突出：
- 这是用哪个算法训练的（`_name`）
- 算法出自哪篇论文（`_paper`）
- 训练过程日志在哪看（wandb/trackio/comet URL）
- 谁也可以用什么子类属性扩展（`_tag_names`、`_template_file`）

对理解训练 GRPO 没什么用，知道一下就行。
## `GRPOTrainer` 类

`GRPOTrainer` 类也不是一个善茬，`grop_trainer.py` 文件有整整 2822 行。好在很多都是工具函数，或为了工程化做的补丁，我们也直击最重要的两个函数：`_prepare_inputs()` 与 `compute_loss()`.

`GRPOTrainer` 重写了这两个函数，让 RL 循环嵌入了基类 `Trainer` 的训练循环中。而 GRPO 的 rollout、打分与计算 loss 都在这两个函数的内部实现，对 `Trainer` 的训练循环没有侵入式的影响。

鉴于读者一个已经掌握了 RL 的基础算法知识，了解 GRPO 是在做什么，这里就按 GRPO 的几个模块来分类：`rollout → reward → advantage → loss`。其中，rollout、reward、advantage 部分都在 `_prepare_inputs()` 内部完成，而计算 GRPO 的 loss 在 `compute_loss()` 里完成
### `_prepare_input()`

这个函数比较简单，做这么几件事：
- 每 `generate_any` step：
	1. 真正的 rollout 一次（由 `_generate_and_score_completions()` 处理）
	2. 打乱结果顺序（避免同 prompt 的 G 个 completion 落在同一 micro-batch）
	3. 切分成 micro batch，作为 `inputs` 给之前的最内层循环使用
	4. 更新缓存
- 从缓存中读当前 step 要用的那一份

它在工程实现上海做了对于文本和图像的双适配，两种 prompt 都走这个路径，不过我们只关注文本就行。在 eval 时，这个函数不做缓冲，每一次调用都重新由 `_generate_and_score_completions()` 生成。

### `_generate_and_score_completions()` - Rollout 侧

这个函数就复杂了，整整 487 行。它的功能囊括了生成 completions、算 reward 和算 advantage，零零碎碎的。流程是这样的：
1. 预处理 prompts ，多模态与 RL 环境交互适配，格式化
2.  `_generate()` ，tokenize + rollout，生成 completions
3. pad 和 mask 输出，成固定 shape （B，P+C）
4. `_get_per_token_logps_and_entropies()` 算两套 logprobs，之后算 loss 用
	1. old：用当前 $\pi_\theta$ 重跑，用作 importance sampling 基准
	2. ref：用 ref_model 算，给 KL penalty 用
5. 计算 rewards 与 advantage
	1. 调用 `_calculate_rewards()` 得到输出 rewards_per_func 
	2. 先是二维矩阵，`[completion, reward_func]`，对应每个 completion 的多个评价函数值
	3. 在 reward_func 维度，加权求和得到 rewards，维度`[completion]`
	4. 计算 `mean_grouped_reward`，组内求平均，但整个维度还是 `[completion]` 不变
	5. 计算 advantage 并归一化
6. 收集输出、保存 log 等等的杂活

其中 5.5 的计算组内 advantage 方式如下：
```python
advantages = (rewards - mean_grouped_rewards) / (std_rewards + 1e-4)
```
$$
A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \varepsilon}
$$

简单来说，一次 generate_and_score_completions 产出"一批 completion 的全部 RL 训练素材"：token、mask、advantage、参考 logprob、多模态辅料；同时把 reward/采样/漂移的各种指标写进日志管道。它是 GRPO 数据流水线的唯一源头，下游所有 update step 都吃它的输出。

它调用的几个函数在这里直接解释一下，不复杂就不单开章节细讲了：

`_generate()`：调用 vllm 或自定义的 `rollout_func`，生成 completion，可以当黑盒看待

`_calculate_rewards()`：调用由用户注入的 `reward_funcs` 函数，逐 completion 逐 func 打分。Trainer 在这里只负责编排使用，不关心它的具体实现。reward_func 可以是 Python 函数、reward model、远程服务等等。

### `_get_per_token_logps_and_entropies()` 算 logprobs

**回顾一下 logps 咋算的：** 我们实际要算的是“在 $s_t$ 下，做 $a_t$ 动作概率的 log。在 LLM 里：
- **state** $s_t$ = prompt + 已经生成的 tokens $y_{< t}$
- **action** $a_t$ = 下一个 token $y_t$ 

$$ \log \pi_\theta(a_t \mid s_t) = \log \pi_\theta  (y_t \mid x, y_{< t} ) $$

这个函数做了什么：
1. forward 获得 `logits = model(input_ids, attention_mask).logits`
	- `input_ids` shape 是 `[B, prompt_len]` 的，是“具体的数（token id）”
	- `logits` shape 是 `[B, prompt_len + completion_len, vocab_size]` 的，是“分布”
2. logits 对齐与整形
	1. 去掉 `input_ids` 第一个，`logits` 最后一个，对齐 $s_t$ 与 $a_t$
	2. 只保留 completion 段（算 prompt 段没有意义，不关模型的事）
	3. 对 temperature 归一化
3. 算 log_softmax + gather
数学上：
$$\log \pi(y_t \mid \cdot) = \text{logits}[y_t] - \log \sum_v \exp(\text{logits}[v])$$
代码上等价于：

```python
log_probs = logits.log_softmax(dim=-1)              # [B, L, V]
per_token_logps = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
# shape: [B, L]
```
TRL 里有个优化叫 `selective_log_softmax`，直接在目标 token 上算，避免 materialize 整个 `[B, L, V]` 的 log_probs 张量——V 是 ~150k（Qwen2.5），这一步能省大量显存。数学上完全等价：
$$\log \pi(y_t) = z_{y_t} - \text{logsumexp}(z)$$
只需要对每个位置算一次 `logsumexp` 和取一次 `z_{y_t}`。

### `compute_loss()` - Training 侧 

总算看完`_prepare_input()` 了，到第二个大模块，loss 计算。然后其实 `compute_loss()` 是对 `_compute_loss()` 的一个超薄封装，如果 `self.use_liger_kernel` 为 True 则路由到另一个 liger grpo loss 计算，正常的话直接调用的是 `_compute_loss()`。后面就直接讲 `_compute_loss()` 内的东西了。

我们先梳理一下现在手上有哪些东西：
- `prompt_ids`, `prompt_mask`
	- prompt 段 token id 与有效位掩码
	-  `_prepare_inputs()` 的 input 处理环节产生
- `completions_ids`, `completion_mask` 
	- 采样得到的 completion 段与掩码
	- 在 rollout 环节` _generate()` 时生成
- `old_per_token_logps` 
	- 旧策略下每 token 的对数概率
	- 在 rollout 里 `_get_per_token_logps_...()` 算的 $\log \pi_\text{old}(a_t \mid s_t)$ 
- `ref_per_token_logps` 
	- 参考策略下每 token 的对数概率
	- 在 rollout 里 `_get_per_token_logps_...()`算的 $\log \pi_{ref}(a_t \mid s_t)$ 
- `advanages` 
	- group 标准化后的优势
	- 在 rollout 里调用 `_calculate_rewards()` 后计算得到组内 $\hat A_i = (r_i - \bar r)/\sigma_r$

要计算 loss，还缺了一个 `per_token_logps`:
- 当前策略对数概率 $\log \pi_\theta(y_t \mid x, y_{< t})$ 
- 在 `_compute_loss()` 里面自己算
- 还是用 `_get_per_token_logps_and_entropies()`

然后我们再看看需要用到的几个中间量：

**importance ratio** 重要性采样 $r_t$:

$$r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_\text{old}(a_t \mid s_t)} = \exp( \log \pi_\theta - \log \pi_\text{old})$$

根据 PPO 的 clip 计算出来的 Policy Gradient 的 **surrogate loss**：

$$\mathcal{L}^{PG}_t = \min (r_t \hat A_t, \text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) \hat A_t ) $$

**KL 正则项** （这个也是 per_token 的哦），在 `beta==0` 的时候会跳过不算：

$$ 
\mathbb{D}_{\mathrm{KL}}[\pi_\theta \,\|\, \pi_{\mathrm{ref}}] = 
\mathbb{E}_{y \sim \pi_\theta}\!
\left[
\log \frac{\pi_\theta(y)}{\pi_{\mathrm{ref}}(y)}
\right]
$$

但是在 GRPO 中不是用的朴素 $\hat k_1 = \log ( \pi_\theta / \pi_\text{ref})$ （方差大且可能为负），用的是：

$$\hat k_3 = \frac{\pi_\text{ref}}{\pi_\theta} - \log \frac{\pi_\text{ref}}{\pi_\theta} - 1 = e^{-\hat k_1} - (-\hat k_1 + 1) $$

同样无偏（在期望意义下），且 $\geq 0$ ，方差更小。

在这里合并，得到 **per_token_loss**：

$$ loss_t = \mathcal{L}^{PG}_t - \beta \cdot \mathbb{D}_{KL} \mid_t$$

然后是合并 per_token_loss 得到**最终 loss**：但是这里 TRL 给了三种不同实现：
- `"grpo"`：原论文：先 per-sequence 平均，再 per-batch 平均
- `"dr_grpo"`：Dr.GRPO：用固定的 max_completion_length 归一化，去掉长度偏差
- `"dapo"`：也叫 token-level，所有有效 token 一起平均，长 completion 权重更大

随后就是 loss 回传，**backward 算梯度**，更新权重了。从 loss 回传回 weight 的路径：

```text
loss 
 └─ per_token_loss
     └─ coef_1 = exp(per_token_logps - old_per_token_logps.detach())
                     ↑ 梯度只从这里流
                 per_token_logps
                     └─ logits.log_softmax().gather(input_ids)
                         └─ model(input_ids) forward
                             └─ model.parameters()  ← 梯度到这
```
只有在 `_compute_loss()` 里面临时算的 `per_token_logps` 这一路径，其他都是 detach 的或者单一的数值，不参与梯度计算。

## 读后感想

觉得最有意思的，是看完 TRL 代码就知道为什么要做 Rollout / Training 分离了。Rollout 侧看似做了很多的 foward 产出了很多，但都是中间数据，只有最后在训练侧的 `compute_loss()` 里面做的那一次 forward 才真正对应着梯度回传的路径。这是两个看起来都在用同一个模型，但是实际上截然不同的两件事，如果做了 R/T 分离，各司其职，直觉上都能让 RL loop 快很多。

此外，代码风格上，感觉 TRL 特别喜欢 no-op，整个代码处处都有 no-op 来简化代码路径。比如说多模态全部塞进一个函数，函数内部如果发现如果没有图片（是单模态）就直接返回，多模态再处理。又比如说做 completion 数组切片，如果是已经切好的，就自然是一个 no-op。让多模态、图片处理、不同的 rollout 后端支持能在明面上尽量走同一个代码路径，看起来很舒服。