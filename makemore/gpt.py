import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig:
  vocab_size: int = 89 #calculated from the *-valid file
  context: int = 256
  n_embd: int = 128
  n_head: int = 4
  n_layer: int = 4
  n_ff_size: int = 512
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

  def forward(self, x): #[B, T]
    B, T = x.shape
    encoded = self.embed(x) #[B, T, C]
    pos = self.position(self.pos_idx[:T])
    x_res = encoded + pos
    layered = self.norm(self.layers(x_res))
    return self.linear(layered)


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
    self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
    self.linear = nn.Linear(config.n_embd, config.n_embd, bias=False)

  def forward(self, x):
    cat = torch.cat([head(x) for head in self.heads], -1)
    return self.linear(cat)


class Head(nn.Module):
  tril: torch.Tensor
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.head_size = config.head_size
    self.w_q = nn.Linear(config.n_embd, config.head_size, bias=False)
    self.w_k = nn.Linear(config.n_embd, config.head_size, bias=False)
    self.w_v = nn.Linear(config.n_embd, config.head_size, bias=False)
    self.register_buffer("tril", torch.ones([config.context, config.context]).tril())

  def forward(self, x) : # x [B, T, n_embed]
    B, T, C = x.shape
    q: torch.Tensor = self.w_q(x) #[B, T, head_size]
    k: torch.Tensor = self.w_k(x) #[B, T, head_size]
    v: torch.Tensor = self.w_v(x) #[B, T, head_size]
    wei = q @ k.transpose(1, 2) / self.head_size ** 0.5 # [B, T, T]
    masked = wei.masked_fill(self.tril[:T, :T] == 0, -torch.inf)
    soft = F.softmax(masked, -1)
    out =  soft @ v # [B, T, head_size]
    return out


class Tokenizer:
  train_data: torch.Tensor
  val_data: torch.Tensor

  def __init__(self, filepath: str):
    data = open(filepath, encoding="utf-8").read().replace("<|endoftext|>", '\x00')
    vocab = sorted(set(ltr for ltr in data))
    self.vocab_size = len(vocab)

    self.stoi = {s: i for i, s in enumerate(vocab)}
    self.itos = {i: s for s, i in self.stoi.items()}

    data = torch.tensor(self.encode(data))
    n = int(len(data) * 0.9)
    self.train_data, self.val_data = data[:n], data [n:]

  def encode(self, data: str):
    return [self.stoi[ch] for ch in data]

  def decode(self, data: list[int]) -> str:
    return "".join([self.itos[i] for i in data])

  def get_batch(self, split, context_size, batch_size, device):
    d = self.train_data if split == "train"  else self.val_data
    ix = torch.randint(0, len(d) - context_size - 1, (batch_size, ))
    x = torch.stack([d[start:start+context_size] for start in ix])
    y = torch.stack([d[start+1:start+context_size+1] for start in ix])
    return x.to(device), y.to(device)

def main() :
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device {device}")
  max_steps = 20_000
  batch_size = 64
  lr_threshold = int(max_steps * 0.7)

  tokenizer = Tokenizer("C:\\dev\\tutorials\\LLM2\\gpt\\makemore\\data\\TinyStoriesV2-GPT4-valid.txt")
  config = GPTConfig(vocab_size=tokenizer.vocab_size)
  model = GPT(config).to(device)

  train(batch_size, config, device, lr_threshold, max_steps, model, tokenizer)

  torch.save({"model": model.state_dict(), "config": config, "stoi": tokenizer.stoi},
             "gpt_tinystories.pt")


def train(batch_size: int, config: GPTConfig, device: str, lr_threshold: int,
    max_steps: int, model: GPT, tokenizer: Tokenizer):
  opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
  sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_threshold],
                                               gamma=0.1)
  for step in range(max_steps):
    xb, yb = tokenizer.get_batch("train", config.context, batch_size, device)
    logits = model(xb)  # [B,T,V]
    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

    if step % (max_steps // 100) == 0 or step == max_steps - 1:
      with torch.no_grad():
        model.eval()
        xv, yv = tokenizer.get_batch("val", config.context, batch_size, device)
        logits = model(xv)
        val_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size),
                                   yv.view(-1))
        print(
          f"step {step}: Training loss={loss.item()}, validation loss = {val_loss.item()}")
        model.train()


if __name__ == "__main__":
  main()


