# Baby Stories

Self-made and home-trained GPT model capable of producing small child stories. Inspired by
- "makemore" video curse by A. Karpathy
- "Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin
- "How Small Can Language Models Be and Still Speak Coherent English?" by Ronen Eldan and Yuanzhi Li of Microsoft Research, published in 2023 as arXiv:2305.07759

# Features
- BPE Tokenizer
- Transformers with Multi-Head Attention 
- Masking of training data based on the beginning of the story
- Uses bf16 for training to be able to train on a laptop
- Resulting checkpoints can be found on my [Hugging Face](https://huggingface.co/nepoleg/baby-stories) 


# Usage

## Train
- download TinyStories dataset (TinyStoriesV2-GPT4-valid.txt and TinyStoriesV2-GPT4-train.txt)
- Run [tokenization] (tokenize.ipynb) to produce the correspondin validation and training encoded files (.bin) 
- Run [gpt.py](gpt.py) to do training. Use `--proceed` to continue from the saved checkpoint
- Use [metrics.csv](metrics.csv) for run's train/val loss charting 

## Inference
- Run [run-gpt.py](run-gpt.py) for inference from the specified saved checkpoint


# Samples

> Once upon a time, there was a little boy named Tim. Tim loved to play outside with his friends. One day, Tim and his friends went to the park to play.
> At the park, Tim saw a big tree. He wanted to climb it. His friend, Sam, said, "Be careful, Tim!" Tim started to climb the tree. He went up, up, up, until he was very high.
> When Tim got to the top, he saw a big bird. The bird said, "Hello, Tim!" Tim was surprised. He did not know birds could talk. The bird told Tim that it was a magic bird. The bird said, "I can make one wish come true for you." Tim thought for a moment and said, "I wish for a big ice cream cone!" The magic bird made a big ice cream cone appear. Tim and his friends were so happy. They all shared the ice cream and had a great day at the park.


- More samples and evolution history can be found [here](TRAINING_RUNS.md)