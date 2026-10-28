import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime
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
    self.layers = nn.ModuleList([Layer(config) for _ in range(config.n_layer)])
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
    doc = (x == 0).cumsum(dim = 1) #[B, T]
    causal_mask = torch.ones((T, T), dtype=torch.bool, device=x.device).tril() # [T, T]
    mask = ((doc[:, None, :] == doc[:, :, None]) & causal_mask) # masks at the start of the stories (so that stories don't clash) [B, T, T]
    mask = mask[:, None, :, :] #[B, 1, T, T]  - 1 will be scaled to n_head
    encoded = self.embed(x)  #[B, T, C]
    pos = self.position(self.pos_idx[:T])
    x_res = encoded + pos
    for layer in self.layers:
      x_res = layer(x_res, mask)
    normed = self.norm(x_res)
    return self.linear(normed)

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


  def forward(self, x, mask):
    att = x + self.multihead_attention(self.ln1(x), mask)
    return att + self.feed_forward(self.ln2(att))


class FeedForward(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.l1 = nn.Linear(config.n_embd, config.n_ff_size)
    self.gelu = nn.GELU()
    self.l2 = nn.Linear(config.n_ff_size, config.n_embd)

  def forward(self, x):
    return self.l2(self.gelu(self.l1(x)))

class MultiHeadAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config
    self.w_q = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.w_k = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.w_v = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.linear = nn.Linear(config.n_embd, config.n_embd, bias=False)

  def forward(self, x, mask):
    B, T, C = x.shape
    q: torch.Tensor = self.w_q(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    k: torch.Tensor = self.w_k(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    v: torch.Tensor = self.w_v(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    attn = F.scaled_dot_product_attention(q,k,v, is_causal=False, attn_mask=mask) #[B, n_heads, T, head_size]
    attn = attn.transpose(1, 2).contiguous().view(B, T, self.config.n_embd) #[B, T, head_size * n_heads]
    return self.linear(attn)


class BatchProvider:
  VAL_BATCH_N = 20

  def __init__(self, train_path: str, val_path: str, context_size, batch_size, device):
    self.context_size = context_size
    self.batch_size = batch_size
    self.device = device
    self.train_buffer = np.memmap(train_path, dtype=np.uint16, mode="r")
    self.val_batches =  self.create_val_batches(val_path)

  def get_train_batch(self):
    ix = torch.randint(0, len(self.train_buffer) - self.context_size - 1, (self.batch_size, ))
    x = torch.stack([torch.from_numpy(self.train_buffer[start:start+self.context_size].astype(np.int64)) for start in ix])
    y = torch.stack([torch.from_numpy(self.train_buffer[start+1:start+self.context_size+1].astype(np.int64)) for start in ix])
    return x.to(self.device), y.to(self.device)

  def create_val_batches(self, val_path):
    batches = []
    buffer = np.memmap(val_path, dtype=np.uint16, mode="r")
    batch = []
    for window_no in range(self.VAL_BATCH_N * self.batch_size):
      start = window_no * self.context_size
      x=torch.from_numpy(buffer[start:start+self.context_size].astype(np.int64))
      y=torch.from_numpy(buffer[start+1:start+self.context_size+1].astype(np.int64))
      batch.append((x, y))
      if len(batch) == self.batch_size:
        b_x = torch.stack([x for x, _ in batch]).to(self.device)
        b_y = torch.stack([y for _, y in batch]).to(self.device)
        batches.append( (b_x, b_y) )
        batch = []
    return batches

def main() :
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device {device}")

  max_steps = 100_000
  batch_size = 32
  tokenizer = Tokenizer.from_file("data/tokenizer.json")
  config = GPTConfig(vocab_size=tokenizer.get_vocab_size(with_added_tokens=True))
  batcher = BatchProvider("C:\\dev\\tutorials\\LLM2\\gpt\\makemore\\data\\TinyStoriesV2-GPT4-train.bin",
                        "C:\\dev\\tutorials\\LLM2\\gpt\\makemore\\data\\TinyStoriesV2-GPT4-valid.bin", config.context, batch_size, device)
  print(f"{config=}")
  model = GPT(config).to(device)
  train(config, device, max_steps, model, batcher, tokenizer)
  torch.save({"model": model.state_dict(), "config": config},"gpt_final.pt")
  model.eval()
  print(sample(model, tokenizer, device, 500))


@torch.no_grad()
def sample(model: GPT, tokenizer: Tokenizer, device: str, length: int) -> str:
  model.eval()
  start = torch.tensor([tokenizer.encode("<|endoftext|>").ids], device=device)
  decode = tokenizer.decode(model.generate(start, length)[0].tolist(), skip_special_tokens=False)
  model.train()
  return decode


def train(config: GPTConfig, device: str, max_steps: int,
    model: GPT, batcher: BatchProvider, tokenizer: Tokenizer, warmup: int = 1000):
  opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
  sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps - warmup),
  ], milestones=[warmup])
  model.train()
  t0 = time.perf_counter()
  last_saved_loss = 10000.0
  for step in range(max_steps):
    xb, yb = batcher.get_train_batch()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits = model(xb)  # [B,T,V]
      loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

    if step % (max_steps // 100) == 0 or step == max_steps - 1:
      val_loss = calc_val_loss(device, model, batcher)
      lr = sched.get_last_lr()[0]
      log_metrics(step, lr, loss.item(), val_loss)
      print(f"step {step} [{time.perf_counter() - t0:.0f}s] lr={lr:.2e}: Training loss={loss.item()}, validation loss = {val_loss}")
      print(sample(model, tokenizer, device, 200))
      if val_loss < last_saved_loss :
        last_saved_loss = val_loss
        start = time.perf_counter()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "config": config}, "best.pt")
        took = time.perf_counter() - start
        print(f"Saved best in {took*1000:.0f} ms. Loss = {val_loss}")


def log_metrics(step: int, lr: float, train_loss: float, val_loss: float):
  new = not os.path.exists("metrics.csv")
  with open("metrics.csv", "a") as f:
    if new:
      f.write("time,step,lr,train_loss,val_loss\n")
    f.write(f"{datetime.now().isoformat(timespec='seconds')},{step},{lr:.3e},{train_loss:.4f},{val_loss:.4f}\n")


@torch.no_grad()
def calc_val_loss(device: str, model: GPT, batcher: BatchProvider) -> float:
  model.eval()
  losses = []
  for xv, yv in batcher.val_batches:
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits = model(xv)
      losses.append(F.cross_entropy(logits.view(-1, logits.shape[-1]), yv.view(-1)).item())
  model.train()
  return sum(losses) / len(batcher.val_batches)

if __name__ == "__main__":
  main()
