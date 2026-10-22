# Char-level GPT on TinyStories: training runs

All runs: char-level tokens, pre-LN transformer, AdamW, final LayerNorm, no dropout.
Loss is cross-entropy per **char** (nats). Samples drawn at temperature 1.0.

## Summary

| # | Params | Layers × n_embd × heads | Context | Batch | Data | Steps | Best val loss | Time |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.85M | 4 × 128 × 4 | 256 | 64 | valid file, 20M chars (90/10 split) | 20k | ~0.67 (single-batch val, noisy); ~0.70 at end | 21 min (62 ms/step) |
| 2 | 4.9M | 6 × 256 × 8 | 256 | 64 | train file, first 1B chars | 100k | **0.501** (step 97k) | 8.2 h |
| 3 | 5.0M | 6 × 256 × 8 | 1024 | 16 | train file, first 1B chars | — | started, then stopped (no results) | — |
| 4 | 11.5M | 8 × 384 × 8 | 1024 | 16 | train file, first 1B chars, filtered | 100k | **~0.40** (step 95k, from memory); final ~0.42 | ~10 h (~0.36 s/step) |

## Run 1: small model, valid file only

- Config: `vocab 89, context 256, n_embd 128, n_head 4, n_layer 4, n_ff 512` (846,937 params)
- Data: `TinyStoriesV2-GPT4-valid.txt` (22.5 MB), 90/10 train/val split
- LR: 1e-3, ×0.1 at 70% of steps (MultiStepLR)
- Val loss from one batch per eval, so it jumps around by ±0.05
- Loss curve: 0.82 at step 6k, 0.75 at step 14k (LR drop), ~0.70 at end. Plateau, train ≈ val (underfitting)
- Observation: words mostly correct, little sense

Sample (full story via `run_gpt.py`, prompt "Once upon a time"):

> Once upon a time, in a small town, there lived a boy named Tom.
> One day, a little girl cames under the tent the tent under her taxi. She saw something shining light, but one day would fly.
> Mia wanted to make Jane. She told the top of her waiting fun. The top was big and broken shiny and ran away. Mia shaked his wings the teddy bear and wish the television. [...]

### Run 1b: bigger model on the same 20M chars (not finished)

- 4.8M params (6 × 256 × 8), still on the valid file only (~17 epochs planned)
- Stopped in favour of run 2. Mid-training sample at val loss 0.797:

> Once upon a time, in a big garden, there livtle go named Sam open the chirros. The lion nodded to look at the bunny and the rocket was stufff.

## Run 2: 4.9M params, 1B chars of train data

- Config: `vocab 191, context 256, n_embd 256, n_head 8, n_layer 6, n_ff 1024` (~4.9M params)
- Vocab 191: started before the rare-char filter (Chinese text, emoji)
- Data: first 1B chars of `TinyStoriesV2-GPT4-train.txt`, val = full valid file
- LR: warmup 1000 steps to 1e-3, cosine to 1e-4
- Val loss averaged over 20 batches
- Naive per-head attention (before `scaled_dot_product_attention`)
- ~1.6 epochs; train ≈ val throughout, no overfitting
- Last 10k steps: 0.511 → 0.501, close to plateau
- Checkpoints: overwritten by run 4 (no longer available)

Sample (step 90k):

> Once upon a time, there was a little girl named Lily. She had a big sack in her room. She loved to draw and write. One day, Lily wanted to show her friend, Tim. She ran to Tim's house and said, "Look

## Run 4: 11.5M params, context 1024

- Config: `vocab 89, context 1024, n_embd 384, n_head 8, n_layer 8, n_ff 1024` (11,496,025 params, read from `best.pt`)
- Note: `n_ff` is 1024 here, not the usual 4 × n_embd = 1536
- Data: first 1B chars of train, stories with chars outside the valid-file vocab dropped (~0.02%)
- Attention: `F.scaled_dot_product_attention` (FlashAttention), ~2.5× faster than per-head version
- LR: warmup 1000 steps to 1e-3, cosine to 1e-4
- ~1.6 epochs; train ≈ val, no overfitting
- Val loss: 0.434 (59k) → 0.427 (67k) → 0.425 (76k) → 0.417 (82k) → ~0.40 at 95k (best, from memory; console output lost), final ~0.42
- Checkpoints: `best.pt` (step 95k), `gpt_final.pt` (step 100k)
- Observation: grammar and spelling nearly clean; logic still weak ("Jack saw Jack smiling", a lollipop on a tree)

Samples:

> One day, a little girl named Lily went to the park. She saw a big tree with a lollipop on it. Lily wanted to examine the lollipop. She walked up to the tree and said, "Hello, lollipop!" to the tree.

> One day, a little boy named Tim went to play outside. He saw a wet dog named Jack. Tim was happy to see Jack. Tim did not yield Jack words, but he was curious.
> When Tim arrived, Jack saw Jack smiling

### Run 4 `best.pt`: full stories by temperature

Generated with `run_gpt.py` until end-of-text. Every story ends on its own, within the 1024 context.

| Temperature | Grammar | Coherence |
|---|---|---|
| 1.0 | mostly correct | stays on topic, odd logic (a toy snake that talks, "gathering more snakes") |
| 0.8 | correct | clear arc, logic jumps ("I want to ring the bear, but I am too small") |
| 0.6 | correct | mostly sensible; can loop on one idea ("The race was going away") |
| 0.4 | correct, rare slips ("eated") | sensible and simple, small cause-effect gaps (a leaf under the tree rescues the cat) |
| 0.2 | correct | most coherent, with a moral; repetitive, safe phrasing; small contradictions |

**T = 1.0**

> Once upon a time, there was a shy little boy named Tim. He loved to play with his toy snake. One day, the snake said, "Tim, I want to put the snake in the water."
> Tim was scared and hid behind a tree. He asked his mom and dad about the snake. They were very shy, but they did not mind. They thought it was a fun toy.
> They put the snake in the water and went to save the snake. The snake said, "Thank you for helping me." The snake and its friends were happy in their heart. They kept playing and gathering more snakes. The shy little boy felt proud of his actions.

> One day, a chubby cat named Tim went to the park. He liked to play with his ball and run around. Tim would go first, then roll in the grass. His friend, a big dog named Max was already to play next to Tim.
> "I want to go first," said Tim. Max stood on his back legs and then looked at Tim. They walked slow together, having a lot of fun. But then, they heard a little crack. Tim looked around and saw a ball in the corner. He liked to play, so Max played with the ball near the park.
> They took turns going first, trying to reach the top. They laughed and went inside to get a better look. After a fun day, they decided to go back home. Tim and Max were very happy and couldn't wait to come back to the park again.

**T = 0.8**

> Once upon a time, there was a big young girl named Lucy. She loved to solve problems in her mind. One day, she found a lucky video that she wanted to collect. She was very happy and started to play the video.
> But then, the video did not stop. It was so big and pretty! Lucy was sad and did not know what to do. She asked her friend, Tom, to help her. Tom said, "I can try to collect things from the video."
> Tom saw that Lucy was sad and she started to collect things. He was having so much fun collecting things to see! But then, something unexpected happened. The big young girl started to talk! The big young girl said, "I am a magic girl! I can make your toys special and be good." They played together and had a great day at the park.

> Once upon a time, there was a pretty bird named Bob. Bob loved to fly high in the sky and sing. One day, he was flying high in the sky when he saw a big bear. The bear was very friendly and liked to jump.
> Bob wanted to help the bear, so he walked up to the bear and said, "Hi, bear! Do you want to ring for me?" The bear looked at Bob and said, "Yes, I want to ring the bear, but I am too small." Bob was sad but he knew that the bear would be happy to have the friendly bear.
> The next day, Bob and the bear came to the park to ring the bear. The bear said, "You need to be kind and help others. Now, you must be kind to everyone in need." Bob was happy to help the bear. From that day on, the bear always helped Bob whenever he could. They became good friends and helped each other in need.

**T = 0.6**

> Once upon a time, there was a strong boy named Tim. Tim had a long toy car. He loved to play with it every day.
> One day, Tim went to the park with his mom. He was so happy! But then, a big dog came and took his toy car. The dog wanted the toy car too. Tim was sad.
> Tim's mom saw him and said, "Don't worry, Tim. We can fix your toy car." She gave Tim a big hug and said, "Thank you, Tim. You are a good friend." Tim felt happy again. The big dog was happy too. They both looked at the toy car and smiled.

> Once upon a time, there was a big race. All the animals wanted to win. The race was closed, and the other animals were very excited.
> The race began. The animals ran as fast as they could. They ran and ran, but the race was too fast. The race was going away. The animals were sad.
> But then, the race started. It was very sad. The race was going away. The big rabbit was sad. The end.

**T = 0.4**

> Once upon a time, there was a big red barn. In the barn, there was a little girl named Lily. Lily loved to play with her toys and eat yummy food. One day, Lily saw a cat stuck in a tree. She wanted to help the cat get down.
> Lily tried to climb the tree, but it was too high. She tried and tried, but she could not reach the cat. Then, she had an idea. She found a big leaf and put it under the tree. The cat was very happy and started to eat the yummy food.
> After eating, Lily saw the cat and the big leaf. She was so happy that she could help. Lily and the cat became best friends. They played and eated together every day. And they lived happily ever after.

> Once upon a time, in a small house, there was a little girl named Amy. She loved to play outside in the sun. One day, she saw a big red ball in the yard. She wanted to play with it.
> Amy tried to get the ball, but it was too high. She tried and tried, but she could not reach it. She felt sad. Her mom saw her and said, "Don't worry, Amy. We can fix it."
> They worked together to fix the ball. When they finished, they put it back in the yard. The ball was so happy to be a big red ball. Amy and her mom went back to playing in the yard. They were happy they found the big red ball.

**T = 0.2**

> Once upon a time, there was a little boy named Tim. Tim loved to play with his toy cars. He had a big box of toy cars that he liked to make them go fast. One day, Tim wanted to play with his cars in the park.
> Tim asked his mom, "Can I play with my cars in the park?" His mom said, "No, Tim. You have to put your cars in the box and make them go fast." Tim was sad, but he listened to his mom.
> The next day, Tim went to the park with his cars. He saw a big tree and thought it would be fun to make them go faster. He put his cars in the box and made them go faster. Tim was happy and played with his cars in the park all day.

> Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved to play with. One day, she went to the park with her mom. They saw a big tree and decided to play near it.
> As they played, Lily saw a small, shiny rock. She wanted to pick it up, but her mom said no. Lily was sad and angry. She thought her mom would not like it. So, she picked up the rock and took it home.
> When Lily got home, her mom saw the rock and said, "Lily, you should not pick up things that are not yours. You should ask for help when you are sad." Lily listened to her mom and went back to playing with her ball. She was happy and played with her ball all day.

## Ideas not tried yet

- Top-k / top-p sampling
- Story-aligned windows (start each window at a story start) + keep only stories ≤ 1024 chars (88% of stories, 78% of chars)
- BPE tokenizer (~4× more story per context, no spelling to learn)
- Bigger model, best combined with BPE
