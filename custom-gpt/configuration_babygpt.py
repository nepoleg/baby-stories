from dataclasses import dataclass

from transformers import PreTrainedConfig

class BabyGPTConfig(PreTrainedConfig):
  model_type = "baby-gpt"
  def __init__(self,
    vocab_size: int,
    context: int = 512,
    n_embd: int = 384,
    n_head: int = 6,
    n_layer: int = 12,
    n_ff_size: int = 1536,
    **kwargs
  ):
    self.context = context
    self.n_embd = n_embd
    self.n_head = n_head
    self.n_layer = n_layer
    self.n_ff_size = n_ff_size
    self.vocab_size = vocab_size
    super().__init__(**kwargs)

  @property
  def head_size(self) -> int:
    return self.n_embd // self.n_head
