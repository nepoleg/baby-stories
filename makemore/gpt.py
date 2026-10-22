import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class GPTConfig:
  vocab_size: int = 89 #calculated from the *-valid file
  context: int = 1024
  n_embd: int = 384
  n_head: int = 8
  n_layer: int = 8
  n_ff_size: int = 1024
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
  def generate(self, idx, max_new_tokens, stop=None, temperature = 1.0): #idx [B, T]
    for _ in range(max_new_tokens):
      logits = self(idx[:, -self.config.context:])
      probs = F.softmax(logits[:, -1, :] / temperature, -1) #[B, V]
      idx = torch.cat([idx, torch.multinomial(probs, 1)], 1)
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


class Tokenizer:
  train_data: torch.Tensor
  val_data: torch.Tensor

  def __init__(self, train_path: str, val_path: str, max_chars: int):
    val_text = self.read(val_path, -1)
    vocab = sorted(set(val_text))
    allowed = set(vocab)
    train_text = "\x00".join(s for s in self.read(train_path, max_chars).split("\x00")
                             if set(s) <= allowed)
    assert len(vocab) <= 256, f"vocab of {len(vocab)} chars does not fit uint8"
    self.vocab_size = len(vocab)

    self.stoi = {s: i for i, s in enumerate(vocab)}
    self.itos = {i: s for s, i in self.stoi.items()}

    self.train_data = self.encode_tensor(train_text)
    self.val_data = self.encode_tensor(val_text)

  @staticmethod
  def read(path: str, max_chars: int) -> str:
    with open(path, encoding="utf-8") as f:
      return f.read(max_chars).replace("<|endoftext|>", '\x00')

  def encode_tensor(self, text: str, chunk: int = 10_000_000) -> Tensor:
    return torch.cat([torch.tensor(self.encode(text[i:i + chunk]), dtype=torch.uint8)
                      for i in range(0, len(text), chunk)])

  def encode(self, data: str):
    return [self.stoi[ch] for ch in data]

  def decode(self, data: list[int]) -> str:
    return "".join([self.itos[i] for i in data])

  def get_batch(self, split, context_size, batch_size, device):
    d = self.train_data if split == "train"  else self.val_data
    ix = torch.randint(0, len(d) - context_size - 1, (batch_size, ))
    x = torch.stack([d[start:start+context_size] for start in ix])
    y = torch.stack([d[start+1:start+context_size+1] for start in ix])
    return x.to(device).long(), y.to(device).long()

def main() :
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device {device}")

  max_steps = 100_000
  batch_size = 16

  t = time.perf_counter()
  tokenizer = Tokenizer("C:\\dev\\tutorials\\LLM2\\gpt\\makemore\\data\\TinyStoriesV2-GPT4-train.txt",
                        "C:\\dev\\tutorials\\LLM2\\gpt\\makemore\\data\\TinyStoriesV2-GPT4-valid.txt",
                        max_chars=1_000_000_000)
  print(f"Loaded {len(tokenizer.train_data):,} train / {len(tokenizer.val_data):,} val chars, "
        f"vocab {tokenizer.vocab_size} in {time.perf_counter() - t:.0f}s")
  config = GPTConfig(vocab_size=tokenizer.vocab_size)
  print(f"{config=}")
  model = GPT(config).to(device)

  train(batch_size, config, device, max_steps, model, tokenizer)

  torch.save({"model": model.state_dict(), "config": config, "stoi": tokenizer.stoi},
             "gpt_final.pt")

  model.eval()
  print(sample(model, tokenizer, device, 500))


@torch.no_grad
def sample(model: GPT, tokenizer: Tokenizer, device: str, length: int) -> str:
  model.eval()
  start = torch.tensor([tokenizer.encode("\x00")], device=device)
  decode = tokenizer.decode(model.generate(start, length)[0].tolist())
  model.train()
  return decode


def train(batch_size: int, config: GPTConfig, device: str, max_steps: int,
    model: GPT, tokenizer: Tokenizer, warmup: int = 1000):
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
    logits = model(xb)  # [B,T,V]
    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

    if step % (max_steps // 100) == 0 or step == max_steps - 1:
      val_loss = calc_val_loss(batch_size, config, device, model, tokenizer)
      print(f"step {step} [{time.perf_counter() - t0:.0f}s] lr={sched.get_last_lr()[0]:.2e}: Training loss={loss.item()}, validation loss = {val_loss}")
      if val_loss < last_saved_loss :
        last_saved_loss = val_loss
        start = time.perf_counter()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "config": config, "stoi": tokenizer.stoi}, "best.pt")
        took = time.perf_counter() - start
        print(f"Saved best in {took*1000:.0f} ms. Loss = {val_loss}")
      if step % (max_steps // 10) == 0:
        print(sample(model, tokenizer, device, 200))

@torch.no_grad
def calc_val_loss(batch_size: int, config: GPTConfig, device: str, model: GPT,
    tokenizer: Tokenizer, n=20) -> float:
  model.eval()
  losses = []
  for _ in range(n):
    xv, yv = tokenizer.get_batch("val", config.context, batch_size, device)
    logits = model(xv)
    losses.append(F.cross_entropy(logits.view(-1, tokenizer.vocab_size), yv.view(-1)).item())
  model.train()
  return sum(losses) / n

if __name__ == "__main__":
  main()
