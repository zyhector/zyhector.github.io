import torch
from model import *

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
train_data = data[:n].to(device)
val_data = data[n:].to(device)


# 辅助函数
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - Hyperparameters.block_size, (Hyperparameters.batch_size,))
    x = torch.stack([data[i:i+Hyperparameters.block_size] for i in ix])
    y = torch.stack([data[i+1:i+Hyperparameters.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


model = NanoGPT(Hyperparameters()).to(device)
torch.set_float32_matmul_precision('high')
model = torch.compile(model)

train_iters = 5000
eval_interval = 500
learning_rate = 2e-4
eval_iters = 100


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

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(
    decode(
        model.generate(context, max_new_tokens=1000)[0].cpu().tolist()
    )
)