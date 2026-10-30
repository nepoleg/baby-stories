# Baby Stories

Self-made and home-trained GPT model capable of producing small child stories. Inspired by
- "makemore" video curse by A. Karpathy
- "Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin
- "How Small Can Language Models Be and Still Speak Coherent English?" by Ronen Eldan and Yuanzhi Li of Microsoft Research, published in 2023 as arXiv:2305.07759

# Features
- BPE Tokenizer
- Transformers with Multi-Head Attention with causual masking and masking based on the star of the story
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

> One day, a little boy named Tim went to play outside. He saw a big tree with a hole in it. Tim was very curious, and wanted to see what was inside. He looked into the hole and saw something shiny.
> Tim wanted to find out what the shiny thing was. He thought it might have a special power to make things better. He tried to lift a big rock, but it was too heavy. He tried to shake it, but it was too strong. Tim felt sad that he could not make things good.
> Then, Tim had an idea. He found a long stick and used it to lift the shiny thing out of the hole. The shiny thing was now a beautiful necklace! Tim was so happy that he danced around and played with his new treasure. The shiny things made his day even more fun.

> Once upon a time, there was a big, healthy bull. He lived on a farm with many animals. The bull was very strong and fast. He had a lot of friends on the farm.
> One day, the bull was playing with his friend, a little bird. They were running and flying. Suddenly, the bull slipped on a rock. The little bird helped the bull get up. They both laughed and continued to play.
> After playing, they went to the farmer's house. The farmer was happy to see them. He gave them a big pot of soup. The bull and the little bird ate the soup together. They were happy and full. The end.


- More samples and evolution history can be found [here](TRAINING_RUNS.md)