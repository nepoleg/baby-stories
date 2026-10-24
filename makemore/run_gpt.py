import argparse
import os
import torch
from tokenizers import Tokenizer
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
  tokenizer = Tokenizer.from_file("data/tokenizer.json")

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
      print(f"{mode=}")
      encoded = tokenizer.encode("<|endoftext|>" + args.prompt).ids
      start = torch.tensor([encoded], device=device)
      out = model.generate(start, args.max_length, stop=encoded[0], **mode)[0].tolist()
      print(tokenizer.decode(out))
    temperature -= 0.2

if __name__ == "__main__":
  main()
