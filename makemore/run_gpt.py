import argparse
import os
import torch

from gpt import GPT, GPTConfig  # GPTConfig must be importable to unpickle the checkpoint

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("prompt", nargs="?", default="")
  parser.add_argument("-n", "--max-length", type=int, default=5000)
  parser.add_argument("-c", "--checkpoint", default=os.path.join(os.path.dirname(__file__), "best.pt"))
  args = parser.parse_args()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
  config: GPTConfig = ckpt["config"]
  stoi = ckpt["stoi"]
  itos = {i: s for s, i in stoi.items()}

  model = GPT(config).to(device)
  model.load_state_dict(ckpt["model"])
  model.eval()
  temperature = 1.0

  modes = [
    {"temperature": 1.0, "top_k": 5, "top_p": 0.0},
    {"temperature": 1.0, "top_k": 10, "top_p": 0.0},
    {"temperature": 1.0, "top_k": 0, "top_p": 0.9},
    {"temperature": 1.0, "top_k": 0, "top_p": 0.7},

    {"temperature": 0.8, "top_k": 5, "top_p": 0.0},
    {"temperature": 0.8, "top_k": 10, "top_p": 0.0},
    {"temperature": 0.8, "top_k": 0, "top_p": 0.9},
    {"temperature": 0.8, "top_k": 0, "top_p": 0.7},
  ]

  for mode in modes :
    for _ in range(2):
      eot = stoi["\x00"]
      print(f"{mode=}")
      start = torch.tensor([[eot] + [stoi[ch] for ch in args.prompt]], device=device)
      out = model.generate(start, args.max_length, stop=eot, **mode)[0].tolist()
      print("".join(itos[i] for i in out[1:]).rstrip("\x00"))
    temperature -= 0.2

if __name__ == "__main__":
  main()
