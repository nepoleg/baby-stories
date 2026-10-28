# GPT on TinyStories: training runs

All runs: pre-LN transformer, AdamW, final LayerNorm, no dropout.
Runs 1–4 use char-level tokens; loss is cross-entropy per **char** (nats).
Runs 5–7 use a byte-level BPE tokenizer; loss is per **token**, about 4.14 chars each.
Samples drawn at temperature 1.0 unless noted.

## Best story so far

Run 7 `best.pt`, temperature 1.0, top_k 5:

> One day, a little boy named Tim found a big, hard rock. He thought it was fun to play with the rock. He would roll the rock around and watch it go. Tim saw a girl named Sue. Sue was sad because she had a terrible face.
> Tim said, "Hi, I'm Tim. Why are you sad?" Sue said, "I lost my toy. It was under a rock, and I can't find it." Tim wanted to help Sue. They looked for Sue's toy together. They looked under rocks and behind rocks, but they could not find Sue's toy.
> Suddenly, Tim had an idea. He found a small box under a tree. He opened the box and saw Sue's toy! Sue was so happy. She said, "Thank you, Tim!" They played with the toy all day and became best friends.

Setup → problem → search that ties back to the rock → solution → happy end, with dialogue and consistent characters. One slip: "she had a terrible face".

Run 6 `best.pt`, temperature 0.8, top_k 10:

> Once upon a time, there was a little boy named Tim. Tim had a big, rough toy that he loved a lot. It was a small, red car with a funny texture. Tim played with his toy all day and took his car everywhere he went.
> One day, Tim met a new friend named Sam. Sam had an opinion about his toy car. Sam said they should share their toys. But Tim did not want to share his car. They both wanted to play with the car and the bumpy ball.
> Tim and Sam started to disagree with each other. They both wanted to play with the car, but neither wanted to share. Finally, they decided to share and take turns. Tim's rough toy car was now very special and they both had a fun day playing together.

A real conflict (both want the car), resolved logically (taking turns), characters consistent throughout. Small slips: "big, rough toy… small, red car", a "bumpy ball" from nowhere.

Run 5 best, training sample at step 99,999, temperature 1.0, no top-k/top-p:

> Once upon a time, there was a little girl named Lucy. She loved to balance on one leg. One day, Lucy felt troubled because she could not balance well. Her mommy told her, "Don't worry, sweetie. You will try to find a good way to balance."
> Lucy went to a small hole in the fence. At the hole, she found a beautiful veil. The veil was red and sparkled in the sun. Lucy was very happy. She put the veil on her shoulder and started to balance. "Wow, perfect!" she said.
> Emma felt proud of herself. She ran around the village, looking for the prettiest things around her. All her friends also wanted to play with Lucy. They all ran around, jumped, and laughed. Lucy was not troubled anymore. She made everyone happy with her beautiful veil.

Problem → found object → solution → happy end. One slip: "Emma" instead of Lucy.

Best char-level story, run 4 `best.pt`, temperature 0.8, top_p 0.7:

> Once upon a time, there was a big tree with many leaves. A little boy named Tim loved to play near the tree. He would swing high up in the sky and feel the wind on his face.
> One day, Tim saw a small bird stuck in the tree. The bird was sad and scared. Tim wanted to help the bird, so he used his strong legs to reach the bird. The bird was happy and said, "Thank you, Tim!"
> Tim felt good for helping the bird. He shared his leaves with the bird. They became good friends and played together every day. Tim learned that helping others can make you feel happy too.

## Summary

| # | Params | Layers × n_embd × heads | Context | Batch | Data | Steps | Best val loss | Time |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.85M | 4 × 128 × 4 | 256 | 64 | valid file, 20M chars (90/10 split) | 20k | ~0.67 (single-batch val, noisy); ~0.70 at end | 21 min (62 ms/step) |
| 2 | 4.9M | 6 × 256 × 8 | 256 | 64 | train file, first 1B chars | 100k | **0.501** (step 97k) | 8.2 h |
| 3 | 5.0M | 6 × 256 × 8 | 1024 | 16 | train file, first 1B chars | — | started, then stopped (no results) | — |
| 4 | 11.5M | 8 × 384 × 8 | 1024 | 16 | train file, first 1B chars, filtered | 100k | **~0.40** (step 95k, from memory); final ~0.42 | ~10 h (~0.36 s/step) |
| 5 | 9.0M | 8 × 256 × 4, BPE 10k, tied | 512 tokens | 16 | full train file, 539M tokens | 100k | **1.405 / token ≈ 0.34 / char** (step 89k); final 1.43 | 7.0 h (~0.25 s/step) |
| 6 | 18.2M | 8 × 384 × 6, BPE 10k, tied | 512 tokens | 32 | full train file, 539M tokens | 100k | **1.340 / token ≈ 0.324 / char** (step 93k); final 1.37 | 12.5 h (~0.45 s/step) |
| 7 | 25.3M | 12 × 384 × 6, BPE 10k, tied, bf16 | 512 tokens | 32 | full train file, 539M tokens | 100k | **1.307 / token ≈ 0.316 / char** (step 90k); final 1.31 | 5.9 h (~0.21 s/step) |

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
- Checkpoint: `best.pt` (step 95k), in git at `1e9cf44`
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

### Run 4 `best.pt`: top-k and top-p sampling

Temperature is applied before the top-k / top-p cut. Two stories per setting, so the comparison is a strong hint, not a measurement.

| Temperature | Sampling | Result |
|---|---|---|
| 1.0 | top_k 5 / 10 | barely helps; k=10 still invents words ("Spare mention something you can toke") |
| 1.0 | top_p 0.9 | coherent, odd details |
| 1.0 | top_p 0.7 | coherent, varied |
| 0.8 | top_k 5 / 10 | fluent, odd logic ("ate their caves") |
| 0.8 | top_p 0.9 | coherent, leans on the stock "something unexpected happened… started to talk!" plot |
| 0.8 | **top_p 0.7** | **best**: clean arc, fitting moral (see "Best story so far") |
| 0.6 | top_p 0.7 / 0.9 | worse: repetitive loops, close to greedy |

Findings:
- Top-p beats top-k at char level. The number of plausible chars changes a lot by position (few inside a word, many at a word start), so a fixed k is too loose in one place and too strict in another.
- Top-p replaces a low temperature; it does not combine with one. T=0.6 + top-p sharpens the distribution twice and the text degenerates into loops (Holtzman et al., 2019).
- Remaining errors are logic, not words. Decoding can't fix those; only a better model can.
- Default for now: **T = 0.8–1.0, top_p = 0.7–0.8**.

**T = 1.0, top_k 10** (invented words):

> The biggest shape then said, "Spare mention something you can toke anything from the zoo." Suddenly, she began to take an eye.

**T = 1.0, top_p 0.7:**

> Once upon a time, there was a sleepy cat named Tom. Tom liked to take a nap in the park. He would sleep all day long. One day, Tom found a shiny coin on the ground. He was very happy.
> Tom wanted to show his friend, a dog named Max. Tom ran to Max's house with the coin. Max was very sleepy and called Tom. Tom said, "Max, let's play a game!" Max wagged his tail and they played the game. They took turns sleeping and had a lot of fun.
> After a while, Tom and Max got tired. They decided to take a nap. When they woke up, they were both very happy. They said goodbye to each other and went to sleep. Tom and Max had a great day at the park.

**T = 0.8, top_p 0.9:**

> One day, a boy named Tim found a big box of toys. He wanted to see what was inside. He tried to open the box, but it was very hard. He pushed and pulled, but the box would not open. Tim was sad.
> Then, a friendly bird flew down and said, "I can help you open the box!" The bird flew around and pushed Tim out of the box. The bird said, "Thank you, Tim! You are very kind." Tim was happy to help. He pushed the box to the tree and went back to play.
> As they played, something unexpected happened. A little bug came out of the box! The bug said, "I live in this box and we can play together!" Tim and the bird were surprised. They all played together and had a fun day.

**T = 0.6, top_p 0.9** (degenerate loop):

> Once upon a time, there was a big boat. The boat had a long tail. The tail was far away. The boat lived in a harbor. The harbor was in a distant place.
> One day, the boat saw a big boat. The boat wanted to sail on the water. The boat sailed on the water. The boat was happy.
> The boat saw a little fish. The fish sailed on the boat. The boat was happy. The boat sailed on the water. The fish sailed on the water. The boat was not sad anymore.

## Run 5: BPE tokenizer, 9.0M params

- Tokenizer: `ByteLevelBPETokenizer` trained on the train file, `vocab_size=10_000`, special token `<|endoftext|>`; files pre-encoded to `uint16` `.bin`, read with `np.memmap`
- Config: `vocab 10000, context 512, n_embd 256, n_head 4, n_layer 8, n_ff 1024` (~9.0M params: 6.3M transformer, 2.56M tied embedding, 0.13M position)
- Weight tying: `linear.weight = embed.weight`, re-initialized with std 0.02. Without the re-init, step-0 loss was ~160 and the model repeated `<|endoftext|>`; with it, 9.23 ≈ ln(10000)
- Data: full train file (539M tokens ≈ 2.2B chars), ~1.5 epochs; val = full valid file (5.4M tokens), loss averaged over 20 batches
- LR: warmup 1000 steps to 1e-3, cosine to 1e-4
- Val loss: 4.65 (1k) → 2.28 (10k) → 1.81 (20k) → 1.64 (30k) → 1.56 (40k) → 1.51 (50k) → 1.47 (60k) → 1.45 (70k) → 1.43 (80k) → **1.405 (89k, best)** → 1.43 (100k)
- Plateau from ~75k; train ≈ val throughout (capacity-limited, no overfitting)
- vs run 4: **~15% lower loss per char** (0.34 vs 0.40) with fewer params and a smaller transformer (6.3M vs 11M)
- Checkpoint: `best.pt` (step 89k), in git at `4372c9c`
- Observation: grammar clean; errors are logic and entity mix-ups ("the ball wanted to play with the ball", Sue speaking as the cactus), plus repetition ("forgive and forgive")

Sample (step 99,999):

> One day, a little boy named Tim went for a walk. He saw a green plant that looked like a cactus. The cactus was green and spiky. Tim was not scared of the cactus.
> He saw a girl named Sue. Sue was playing with the cactus too. They were having fun. But then the cactus moved! Tim and Sue were scared. Sue said, "Don't be scared, Tim. I am a cactus. I live in the cactus. I am a nice cactus."
> Tim and Sue were happy. They played with the cactus every day. They had a lot of fun. The cactus made them feel brave. And they became good friends.

Sampling (two stories per setting): T 0.8 with top_p 0.7–0.9 gave the most coherent stories; top_k 5 at T 1.0 looped the most ("They hope… They hope…").

## Run 6: wider BPE model, 18.2M params

- Config: `vocab 10000, context 512, n_embd 384, n_head 6, n_layer 8, n_ff 1536` (~18.2M params: 14.2M transformer, 3.84M tied embedding, 0.2M position; matches `gpt_final.pt` size)
- Batch 32 (run 5: 16), so 16k tokens per step; 1.64B tokens total ≈ 3.0 epochs
- Same tokenizer, data, weight tying and LR schedule as run 5
- Val loss: 2.13 (10k) → 1.74 (20k) → 1.60 (30k) → 1.51 (40k) → 1.45 (50k) → 1.42 (60k) → 1.37 (70k) → 1.36 (80k) → **1.340 (93k, best)** → 1.37 (100k)
- Averages over 10 evals (val noise ±0.02): 70–79k ~1.380, 80–89k ~1.369, 90–100k ~1.359. Plateau; continuing would gain ~0.01–0.03
- Train ≈ val throughout, even at 3 epochs: still capacity-limited
- vs run 5: −0.065 per token (−0.016 per char) for 2× params
- Checkpoint: `best.pt` (step 93k), in git at `b17cdac`

### Run 6 `best.pt`: sampling

Two stories per setting, via `run_gpt.py`.

| Temperature | Sampling | Result |
|---|---|---|
| 1.0 | top_k 5 / 10 | coherent; k=10 mixes up characters (cars named Tim, "a big hand caught in a tree") |
| 1.0 | top_p 0.9 | coherent, imaginative (a magic zipper that grants a wish) |
| 1.0 | top_p 0.7 | coherent, leans on the stock "the tree began to talk!" twist |
| 0.8 | top_k 5 | clean; endings flip logic (Spot "glad he could help his friend" who helped him) |
| 0.8 | **top_k 10** | **best**: real conflict and resolution (see "Best story so far") |
| 0.8 | top_p 0.9 | good tension (Mandy and the raft), weak ending: disobeying gets rewarded |
| 0.8 | top_p 0.7 | worse: loops ("Tim was happy. He played with the rock every day" twice) |

Findings:
- Top-k works with BPE. At char level it produced invented words; with BPE every token is a real word piece, so k=5–10 gives clean text. The run 4 finding "top-p beats top-k" does not carry over.
- T 0.8 + top_p 0.7, the best char-level setting, is too strict for BPE and loops. Default now: **T 0.8, top_k 10 or top_p 0.9**.
- Most common error: endings that contradict the story (thanking Mom "for helping me catch the bug" after it flew away).

## Run 7: deeper BPE model, 25.3M params, bf16

- Config: `vocab 10000, context 512, n_embd 384, n_head 6, n_layer 12, n_ff 1536` (~25.3M params)
- bf16 autocast for the training and validation forward passes: 1.49 → 0.25 s/step in a benchmark, 5.6 → 4.2 GiB; no GradScaler needed
- Same batch, tokenizer, data, weight tying and LR schedule as run 6 (~3.0 epochs)
- Val loss: 2.01 (10k) → 1.59 (20k) → 1.49 (30k) → 1.41 (40k) → 1.37 (50k) → 1.34 (60k) → 1.32 (70k) → 1.34 (80k) → **1.307 (90k, best)** → 1.31 (100k)
- Averages over 10 evals: 71–80k ~1.332, 81–90k ~1.314, 91–100k ~1.321. Plateau from ~80k
- Train ≈ val throughout: still capacity-limited
- vs run 6: −0.04 per token (−0.01 per char) on the averages, in less than half the time
- Checkpoint: `best.pt` (step 90k), in git at `bf3c8e3`

### Run 7 `best.pt`: sampling

Two stories per setting, via `run_gpt.py`.

| Temperature | Sampling | Result |
|---|---|---|
| 1.0 | **top_k 5** | **best**: clean, full arcs (see "Best story so far"); the stones story repeats "big, round stone" |
| 1.0 | top_k 10 | coherent; hide-and-seek swaps roles ("Max, you count!"), the old-man story goes nowhere |
| 1.0 | top_p 0.9 | long; loops ("He does not know…") and swaps animals (bird / bee / dog) |
| 1.0 | top_p 0.7 | clean and simple (the beach, Tim's cooler) |
| 0.8 | top_k 5 | fluent; the park-ranger story ends in a contradiction ("wish they had not been brave") |
| 0.8 | top_k 10 | good (Tom and Sue build a house of blocks); the necklace story loops and mixes up Lily and Ben |
| 0.8 | top_p 0.9 | short and flat; odd logic (feeding a bug makes it "a good snack") |
| 0.8 | top_p 0.7 | worst: loops ("She told them they were brave and good" ×3); one clean short story (Mr. Bear) |

Findings:
- Top-k 5–10 stays the best choice with BPE. A stronger model also handles T 1.0 well.
- T 0.8 + top_p 0.7 loops again, as in run 6.
- Fewer word-level slips than run 6. The main errors are still logic: role swaps, endings that contradict the story.

## Ideas not tried yet

- Story-aligned windows (start each window at a story start) + keep only stories ≤ 1024 chars (88% of stories, 78% of chars)
- Use the bf16 speedup for a bigger model: n_embd 512, n_head 8, n_layer 12 (~42M params)
- Shorter run or higher LR floor: runs 5–7 gained little after step ~75–80k
