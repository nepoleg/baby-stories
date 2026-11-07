import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import SequentialLR

from babygpt_model.configuration_babygpt import BabyGPTConfig
from babygpt_model.modeling_babygpt import BabyGPT

METRICS_CSV = "metrics.csv"


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

class GPTConfig:
  'legacy for compatibility with earlier checkpoints'


def main() :
  p = argparse.ArgumentParser(description="Train GPT")
  p.add_argument("--proceed", action="store_true", help="Proceed training from the saved checkpoint")
  p.add_argument("-c", "--checkpoint", default="best.pt")
  args = p.parse_args()

  device = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"Using device {device}")

  batch_size = 32
  warmup: int = 1000
  max_steps = 100_000

  tokenizer = Tokenizer.from_file("data/tokenizer.json")
  config = BabyGPTConfig(vocab_size=tokenizer.get_vocab_size(with_added_tokens=True))
  ckpt = None

  if args.proceed:
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, GPTConfig): #old checkpoints saved the whole config object
      config = BabyGPTConfig(**vars(cfg))
    else : # in newer version it is just a dict
      config  = BabyGPTConfig.from_dict(cfg)
  else:
    if os.path.exists(METRICS_CSV):
      os.replace(METRICS_CSV, f"metrics_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(METRICS_CSV, "a") as f:
       f.write("time,step,lr,train_loss,val_loss,tps\n")

  batcher = BatchProvider("data/TinyStoriesV2-GPT4-train.bin",
                          "data/TinyStoriesV2-GPT4-valid.bin", config.context, batch_size, device)
  print(f"{config=}")

  model = BabyGPT(config).to(device)
  start_step = 0
  opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
  last_saved_loss = 10000.0
  sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps - warmup),
  ], milestones=[warmup])
  if args.proceed:
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    start_step = ckpt["step"] + 1
    last_saved_loss = ckpt.get("loss", 1.277) #replace default with 10000 after this run

  train(config, device, model, batcher, tokenizer, opt, sched, max_steps, start_step, last_saved_loss)
  # torch.save({"model": model.state_dict(), "config": config},"gpt_final.pt") # we are saving best already... who cares for the last
  model.eval()
  print(sample(model, tokenizer, device, 500))


@torch.no_grad()
def sample(model: BabyGPT, tokenizer: Tokenizer, device: str, length: int) -> str:
  model.eval()
  start = torch.tensor([tokenizer.encode("<|endoftext|>").ids], device=device)
  decode = tokenizer.decode(model.generate(start, length)[0].tolist(), skip_special_tokens=False)
  model.train()
  return decode


def train(config: BabyGPTConfig, device: str, model: BabyGPT, batcher: BatchProvider, tokenizer: Tokenizer,
    opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.SequentialLR, max_steps: int, start_step: int, last_saved_loss: float):
  model.train()
  run_start = time.perf_counter()
  for step in range(start_step, max_steps):
    is_validation_step =  step % (max_steps // 100) == 0 or step == max_steps - 1
    if is_validation_step:
      if device == "cuda":
        torch.cuda.synchronize()
      step_start = time.perf_counter()
    xb, yb = batcher.get_train_batch()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits = model(xb)  # [B,T,V]
      loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

    if is_validation_step:
      if device == "cuda": torch.cuda.synchronize()
      step_end = time.perf_counter()
      val_loss = calc_val_loss(device, model, batcher)
      lr = sched.get_last_lr()[0]
      tps = config.context * batcher.batch_size / (step_end - step_start)
      log_metrics(step, lr, loss.item(), val_loss, tps)

      print(f"step {step} [{time.perf_counter() - run_start:.0f}s] lr={lr:.2e}: Training loss={loss.item()}, validation loss = {val_loss}")
      print(sample(model, tokenizer, device, 200))
      if val_loss < last_saved_loss :
        last_saved_loss = val_loss
        save_best_checkpoint(config, model, opt, sched, step, val_loss)


def save_best_checkpoint(config: BabyGPTConfig, model: BabyGPT, opt: Optimizer,
    sched: SequentialLR, step: int, val_loss: float):
  try:
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step,
                "config": config.to_dict(), "loss": val_loss}, "best.pt.tmp")
    os.replace("best.pt.tmp", "best.pt")
    print(f"Saved best with val_loss = {val_loss}")
  except Exception as e:
    print(f"Best Save failed, continuing: {e}")


def log_metrics(step: int, lr: float, train_loss: float, val_loss: float, tps: float):
  with open(METRICS_CSV, "a") as f:
    f.write(f"{datetime.now().isoformat(timespec='seconds')},{step},{lr:.3e},{train_loss:.4f},{val_loss:.4f},{tps:.4f}\n")


@torch.no_grad()
def calc_val_loss(device: str, model: BabyGPT, batcher: BatchProvider) -> float:
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
