import torch

#-------- Model
from babygpt_model.modeling_babygpt import BabyGPTModel
from babygpt_model.configuration_babygpt import BabyGPTConfig

class GPTConfig:
  pass
ckpt = torch.load("best.pt", weights_only=False)
cfg = ckpt["config"]
conf: BabyGPTConfig
if isinstance(cfg, GPTConfig): #old checkpoints saved the whole config object
  conf = BabyGPTConfig(**vars(cfg))
else : # in newer version it is just a dict
  conf = BabyGPTConfig.from_dict(cfg)

config = BabyGPTConfig(conf.vocab_size, conf.context, conf.n_embd, conf.n_head, conf.n_layer, conf.n_ff_size,
                       bos_token_id=0, eos_token_id=0, pad_token_id=0, i_hate_python=True, tie_word_embeddings=True)

model = BabyGPTModel(config)
model.model.load_state_dict(ckpt["model"], strict=False)
model.generation_config.use_cache = False

BabyGPTConfig.register_for_auto_class()
BabyGPTModel.register_for_auto_class("AutoModelForCausalLM")

model.save_pretrained("custom-gpt")


# ------- Tokenizer --------
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json", bos_token="<|endoftext|>", eos_token="<|endoftext|>")
tokenizer.save_pretrained("custom-gpt")