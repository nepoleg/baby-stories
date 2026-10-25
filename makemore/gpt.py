import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tokenizers import Tokenizer
import numpy as np


@dataclass(frozen=True)
class GPTConfig:
  vocab_size: int = 89 #calculated from the *-valid file
  context: int = 512
  n_embd: int = 384
  n_head: int = 6
  n_layer: int = 12
  n_ff_size: int = 1536
  dropout: float = 0.1

  @property
  def head_size(self) -> int:
    return self.n_embd // self.n_head

class GPT(nn.Module):
  pos_idx: torch.Tensor

  def __init__(self, config: GPTConfig):
    super().__init__()
    self.layers = nn.Sequential(*[Layer(config) for _ in range(config.n_layer)])
    self.embed = nn.Embedding(config.vocab_size, config.n_embd)
    self.position = nn.Embedding(config.context, config.n_embd)
    self.linear = nn.Linear(config.n_embd, config.vocab_size)
    self.linear.weight = self.embed.weight # Vector's tying
    nn.init.normal_(self.embed.weight, std=0.02)
    self.register_buffer("pos_idx", torch.tensor([i for i in range(config.context)]))
    self.norm = nn.LayerNorm(config.n_embd)
    self.config = config

  def forward(self, x): #[B, T]
    B, T = x.shape
    encoded = self.embed(x) #[B, T, C]
    pos = self.position(self.pos_idx[:T])
    x_res = encoded + pos
    layered = self.norm(self.layers(x_res))
    return self.linear(layered)

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, stop=None, temperature = 1.0, top_k = 0, top_p = 0.0): #idx [B, T]
    assert idx.shape[0] == 1 #only uses non-batched 0 dimension.
    for _ in range(max_new_tokens):
      logits: torch.Tensor = self(idx[:, -self.config.context:])[0, -1] # [V]
      if top_k > 0:
        low_k = logits.sort(dim=-1, descending=True).indices[top_k:]
        logits[low_k] = -torch.inf
      elif top_p > 0.0:
        srt = (logits  / temperature).softmax(0).sort(dim=-1, descending=True)
        running_sum = srt.values.cumsum(0) - srt.values
        logits[srt.indices[running_sum > top_p]] = -torch.inf

      probs  = F.softmax(logits / temperature, -1)
      idx = torch.cat([idx, torch.multinomial(probs, 1).unsqueeze(0)], 1)
      if stop is not None and (idx[:, -1] == stop).all():
        break
    return idx


class Layer(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.multihead_attention = MultiHeadAttention(config)
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.feed_forward = FeedForward(config)
    self.ln2 = nn.LayerNorm(config.n_embd)


  def forward(self, x):
    att = x + self.multihead_attention(self.ln1(x))
    return att + self.feed_forward(self.ln2(att))


class FeedForward(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.l1 = nn.Linear(config.n_embd, config.n_ff_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(config.n_ff_size, config.n_embd)

  def forward(self, x):
    return self.l2(self.relu(self.l1(x)))

class MultiHeadAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config
    self.w_q = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.w_k = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.w_v = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.linear = nn.Linear(config.n_embd, config.n_embd, bias=False)

  def forward(self, x):
    B, T, C = x.shape
    q: torch.Tensor = self.w_q(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    k: torch.Tensor = self.w_k(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    v: torch.Tensor = self.w_v(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    attn = F.scaled_dot_product_attention(q,k,v, is_causal=True) #[B, n_heads, T, head_size]
    attn = attn.transpose(1, 2).contiguous().view(B, T, self.config.n_embd) #[B, T, head_size * n_heads]
    return self.linear(attn)


class MyTokenizer:
  def __init__(self, train_path: str, val_path: str):
    self.train_buffer = np.memmap(train_path, dtype=np.uint16, mode="r")
    self.val_buffer = np.memmap(val_path, dtype=np.uint16, mode="r")
    self.tokenizer = Tokenizer.from_file("data/tokenizer.json")
    self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)

  def decode(self, data: list[int]) -> str:
    return self.tokenizer.decode(data, skip_special_tokens=False)

  def get_batch(self, split, context_size, batch_size, device):
    buffer = self.train_buffer if split == "train"  else self.val_buffer
    ix = torch.randint(0, len(buffer) - context_size - 1, (batch_size, ))
    x = torch.stack([torch.from_numpy(buffer[start:start+context_size].astype(np.int64)) for start in ix])
    y = torch.stack([torch.from_numpy(buffer[start+1:start+context_size+1].astype(np.int64)) for start in ix])
    return x.to(device).long(), y.to(device).long()

  def encode(self, text: str = "<|endoftext|>"):
    return self.tokenizer.encode(text).ids

def main() :
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device {device}")

  max_steps = 100_000
  batch_size = 32

  tokenizer = MyTokenizer("C:\\dev\\tutorials\\LLM2\\gpt\\makemore\\data\\TinyStoriesV2-GPT4-train.bin",
                        "C:\\dev\\tutorials\\LLM2\\gpt\\makemore\\data\\TinyStoriesV2-GPT4-valid.bin")
  config = GPTConfig(vocab_size=tokenizer.vocab_size)
  print(f"{config=}")
  model = GPT(config).to(device)

  train(batch_size, config, device, max_steps, model, tokenizer)

  torch.save({"model": model.state_dict(), "config": config},
             "gpt_final.pt")

  model.eval()
  print(sample(model, tokenizer, device, 500))


@torch.no_grad()
def sample(model: GPT, tokenizer: MyTokenizer, device: str, length: int) -> str:
  model.eval()
  start = torch.tensor([tokenizer.encode()], device=device)
  decode = tokenizer.decode(model.generate(start, length)[0].tolist())
  model.train()
  return decode


def train(batch_size: int, config: GPTConfig, device: str, max_steps: int,
    model: GPT, tokenizer: MyTokenizer, warmup: int = 1000):
  opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
  sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps - warmup, eta_min=1e-4),
  ], milestones=[warmup])
  model.train()
  t0 = time.perf_counter()
  last_saved_loss = 10000.0
  for step in range(max_steps):
    xb, yb = tokenizer.get_batch("train", config.context, batch_size, device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits = model(xb)  # [B,T,V]
      loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

    if step % (max_steps // 100) == 0 or step == max_steps - 1:
      val_loss = calc_val_loss(batch_size, config, device, model, tokenizer)
      print(f"step {step} [{time.perf_counter() - t0:.0f}s] lr={sched.get_last_lr()[0]:.2e}: Training loss={loss.item()}, validation loss = {val_loss}")
      print(sample(model, tokenizer, device, 200))
      if val_loss < last_saved_loss :
        last_saved_loss = val_loss
        start = time.perf_counter()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "config": config}, "best.pt")
        took = time.perf_counter() - start
        print(f"Saved best in {took*1000:.0f} ms. Loss = {val_loss}")


@torch.no_grad
def calc_val_loss(batch_size: int, config: GPTConfig, device: str, model: GPT,
    tokenizer: MyTokenizer, n=20) -> float:
  model.eval()
  losses = []
  for _ in range(n):
    xv, yv = tokenizer.get_batch("val", config.context, batch_size, device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits = model(xv)
      losses.append(F.cross_entropy(logits.view(-1, tokenizer.vocab_size), yv.view(-1)).item())
  model.train()
  return sum(losses) / n

if __name__ == "__main__":
  main()
