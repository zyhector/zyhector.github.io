import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Hyperparameters:
    batch_size: int = 64
    block_size: int = 256
    vocab_size: int = 65
    d_model: int = 384
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 4 * d_model
    num_blocks: int = 4
    dropout: float = 0.2

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

class MLP(nn.Module):
    def __init__(self, config: Hyperparameters):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: Hyperparameters):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, config: Hyperparameters):
        super().__init__()
        self.config = config
        self.tok_emb_table = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb_table = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)]
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.ln_f = nn.LayerNorm(config.d_model)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_emb_table(idx)  # (B, T, d_model)
        pos_emb = self.pos_emb_table(
            torch.arange(T, device=device)
        )  # (T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)

        x = self.blocks(x)  # (B, T, d_model)
        x = self.ln_f(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()  # 生成文本时不需要计算梯度，会节省内存和计算资源
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # 获取输入的最后 block_size 个字符
            idx_cond = idx[:, -self.config.block_size:]
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