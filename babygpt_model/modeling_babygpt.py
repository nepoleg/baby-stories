import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutput

from .configuration_babygpt import BabyGPTConfig
from transformers import PreTrainedModel, GenerationMixin


class BabyGPT(nn.Module):
  pos_idx: torch.Tensor
  sequence_cache: torch.Tensor | None

  def __init__(self, config: BabyGPTConfig):
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
    self.register_buffer("sequence_cache", None, persistent=False)
    self.cache_size = 0

  def reset(self):
    """Resets the caches and starts a new sequence. To be used for inference with cache=True between the sequences"""
    self.cache_size = 0
    for layer in self.layers:
        layer.reset()

  def forward(self, x: torch.Tensor, use_cache = False): #[B, T]
    B, T = x.shape
    mask: torch.Tensor
    if use_cache:
      if self.cache_size + T > self.config.context : raise Exception("Cache size exceeded")
      if self.sequence_cache is None:
        self.sequence_cache = torch.empty([B, self.config.context], dtype = x.dtype, device=x.device)
      self.sequence_cache[:, self.cache_size:self.cache_size+T] = x
      self.cache_size += T
      mask = self.calculate_mask(self.cache_size, self.sequence_cache[:, :self.cache_size])[:, :, -T:, :]  # we only need last x.T rows (new rows) of the mask in cache mode
    else:
      mask = self.calculate_mask(T, x)
    encoded = self.embed(x)  #[B, T, C]
    pos = self.position(self.pos_idx[self.cache_size-T:self.cache_size] if use_cache else self.pos_idx[:T] )
    x_res = encoded + pos
    for layer in self.layers:
      x_res = layer(x_res, mask, use_cache)
    normed = self.norm(x_res)
    return self.linear(normed)

  def calculate_mask(self, T: int, x: Tensor) -> Tensor:
    doc = (x == 0).cumsum(dim=1)  # [B, T]
    causal_mask = torch.ones((T, T), dtype=torch.bool,
                             device=x.device).tril()  # [T, T]
    mask = ((doc[:, None, :] == doc[:, :, None]) & causal_mask)  # masks at the start of the stories (so that stories don't clash) [B, T, T]
    mask = mask[:, None, :, :]  # [B, 1, T, T]  - 1 will be scaled to n_head
    return mask

  @torch.no_grad()
  def generate(self, idx:torch.Tensor, max_new_tokens, stop=None, temperature = 1.0, top_k = 0, top_p = 0.0): #idx [B, T]
    assert idx.shape[0] == 1 #only uses non-batched 0 dimension.
    self.reset()
    result = idx
    for _ in range(max_new_tokens):
      with torch.autocast(device_type=idx.device.type, dtype=torch.bfloat16): # to save model recompilation on sampling during training
        logits = self(idx[:, -self.config.context:], use_cache=True)[0, -1].float() # [V]

      if top_k > 0:
        low_k = logits.sort(dim=-1, descending=True).indices[top_k:]
        logits[low_k] = -torch.inf
      elif top_p > 0.0:
        srt = (logits  / temperature).softmax(0).sort(dim=-1, descending=True)
        running_sum = srt.values.cumsum(0) - srt.values
        logits[srt.indices[running_sum > top_p]] = -torch.inf

      probs  = F.softmax(logits / temperature, -1)
      idx = torch.multinomial(probs, 1).unsqueeze(0)
      result = torch.cat((result, idx), dim=1)

      if stop is not None and (idx[:, -1] == stop).all():
        break
    return result


class Layer(nn.Module):
  def __init__(self, config: BabyGPTConfig):
    super().__init__()
    self.multihead_attention = MultiHeadAttention(config)
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.feed_forward = FeedForward(config)
    self.ln2 = nn.LayerNorm(config.n_embd)

  def reset(self):
    self.multihead_attention.reset()

  def forward(self, x, mask, use_cache):
    att = x + self.multihead_attention(self.ln1(x), mask, use_cache)
    return att + self.feed_forward(self.ln2(att))


class FeedForward(nn.Module):
  def __init__(self, config: BabyGPTConfig):
    super().__init__()
    self.l1 = nn.Linear(config.n_embd, config.n_ff_size)
    self.gelu = nn.GELU()
    self.l2 = nn.Linear(config.n_ff_size, config.n_embd)

  def forward(self, x):
    return self.l2(self.gelu(self.l1(x)))

class MultiHeadAttention(nn.Module):
  k_cache : torch.Tensor | None
  v_cache : torch.Tensor | None

  def __init__(self, config: BabyGPTConfig):
    super().__init__()
    self.config = config
    self.cache_size = 0
    self.w_q = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.w_k = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.w_v = nn.Linear(config.n_embd, config.head_size * config.n_head, bias=False)
    self.linear = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.register_buffer("k_cache", tensor=None, persistent=False)
    self.register_buffer("v_cache", tensor=None, persistent=False)

  def reset(self):
    self.cache_size = 0

  def forward(self, x, mask, use_cache):
    B, T, C = x.shape
    q: torch.Tensor = self.w_q(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    k: torch.Tensor = self.w_k(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    v: torch.Tensor = self.w_v(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1,2) #[B, n_heads, T, head_size]
    if use_cache:
      if self.cache_size + T > self.config.context: raise Exception("Maximum cache length exceeded")
      if self.k_cache is None:
        self.k_cache = torch.empty((B, self.config.n_head, self.config.context, self.config.head_size), device=x.device, dtype=k.dtype)
        self.v_cache = torch.empty((B, self.config.n_head, self.config.context, self.config.head_size ), device=x.device, dtype=v.dtype)
      self.cache_size += T

      self.k_cache[:, :,self.cache_size-T:self.cache_size, :] = k
      self.v_cache[:, :,self.cache_size-T:self.cache_size, :] = v
      k = self.k_cache[:, :, :self.cache_size, :]
      v = self.v_cache[:, :, :self.cache_size, :]

    attn = F.scaled_dot_product_attention(q,k,v, is_causal=False, attn_mask=mask) #[B, n_heads, T, head_size]
    attn = attn.transpose(1, 2).contiguous().view(B, T, self.config.n_embd) #[B, T, head_size * n_heads]
    return self.linear(attn)


class BabyGPTModel(PreTrainedModel, GenerationMixin):
  config_class = BabyGPTConfig
  _tied_weights_keys = {'model.linear.weight': 'model.embed.weight'}

  def __init__(self, config: BabyGPTConfig):
    super().__init__(config)
    self.model = BabyGPT(config)
    self.post_init()

  def forward(self, input_ids, **kwargs):
    res = self.model.forward(input_ids)
    return CausalLMOutput(logits = res)

