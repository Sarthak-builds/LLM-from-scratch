# Training sarthLLM on Google Colab

Complete walkthrough for pretraining a 124M GPT on WikiText-103 on a single
Colab T4. Every command below is the real thing — no pseudocode.

The two facts that shape everything else:

- **A free Colab session dies at 12 hours.** Official limit. The run below is
  sized to fit inside that, and checkpoints to disk so a preemption costs you
  minutes, not the run.
- **The T4 is `sm_75` and has no bf16.** bf16 tensor cores start at `sm_80`
  (Ampere). Everything here is fp16 autocast with a `GradScaler`. If a tutorial
  tells you to use `torch.bfloat16` on a T4, it is wrong.

---

## 0. Setup (once)

Runtime > Change runtime type > **T4 GPU**. Colab's free tier gives you a T4
with 16 GB VRAM.

```bash
# Colab ships a CUDA build of torch already. Do not reinstall it.
pip install tiktoken datasets
```

If your code is on GitHub, clone it instead of uploading files:

```bash
!git clone https://github.com/Sarthak-builds/llm-from-scratch.git
%cd llm-from-scratch
pip install -e . --no-deps      # --no-deps keeps Colab's torch in place
```

`--no-deps` matters. Without it pip may resolve this project's pinned torch
from the CPU-only index and replace the working GPU build.

The whole dataset is 740 MB on disk (parquet + Arrow cache) plus a ~229 MB
`train.bin`. Colab's disk is roughly 78 GB, so this is comfortable.

---

## 1. Prepare the data (once, ~10 min)

```bash
python -m llm_from_scratch.data.wikitext_prepare --out-dir /content/wikitext
```

Expected output. **The token counts below are estimates derived from the corpus
size, not measured** — the script prints the real numbers, and those are the ones
to trust:

```
downloading Salesforce/wikitext [wikitext-103-raw-v1] split=train ...
  rows: 1,801,350
  joined 536,806,155 chars -> using 536,806,155
  train: ~119,842,103 tokens -> /content/wikitext/train.bin (228.6 MB) in 8.7 min
downloading Salesforce/wikitext [wikitext-103-raw-v1] split=validation ...
  rows: 3,760
  val: ~245,014 tokens -> /content/wikitext/val.bin (0.5 MB) in 0.1 min

total tokens on disk: ~120,087,117
```

Those exact numbers are the ones to trust — the "103M" in WikiText-103 counts
*words*, and GPT-2 BPE is finer-grained, so the token count is higher. The
script prints the real figure rather than assuming it.

**Verify the loaders before you spend GPU hours:**

```bash
python -m llm_from_scratch.data.wikitext_dataloader --data-dir /content/wikitext
```

```
train : tokens  119.84M | windows  234,067 | batches  14,629 | tokens/epoch 119.84M
val   : tokens   65.54K | windows    128   | batches      8 | tokens/epoch 65.54K
input_ids  shape: (16, 512) dtype=torch.int64
target == input shifted by one: True
overlap: stride 512 vs block_size 512 -> none
```

`target == input shifted by one: True` is the one line that must pass. If it
does not, nothing downstream is meaningful.

### Persist the data to Drive

Tokenizing takes ~9 minutes. Re-doing that on every reconnection wastes a
meaningful slice of a 12-hour window, so move the `.bin` files somewhere
durable:

```python
from google.colab import drive
drive.mount('/content/drive')

!cp /content/wikitext/*.bin /content/wikitext/meta.json /content/drive/MyDrive/llm-project/
```

Next session:

```python
!cp /content/drive/MyDrive/llm-project/*.bin /content/drive/MyDrive/llm-project/meta.json /content/wikitext/
```

### Or: a 5-minute smoke test first

```bash
python -m llm_from_scratch.data.wikitext_prepare --out-dir /content/wikitext --shard 0.02
python -m llm_from_scratch.pretraining.train --data-dir /content/wikitext \
    --max-steps 60 --eval-interval 30 --log-interval 10 --ckpt-interval 30
```

2% of the corpus is ~2.4M tokens. Sixty steps is enough to prove the whole
pipeline runs, allocates its memory, checkpoints and resumes. Do this before
committing to a 12-hour run.

---

## 2. Train

```bash
python -m llm_from_scratch.pretraining.train \
    --data-dir /content/wikitext \
    --ckpt-dir /content/checkpoints \
    --drive-dir /content/drive/MyDrive/llm-project/checkpoints \
    --resume
```

The startup banner tells you what you are actually running:

```
torch         : 2.8.0+cu126 (cuda 12.6)
device        : Tesla T4 | 15.8 GB | sm_75  [no bf16: sm_75]
parameters    : 124,000,512 (85,009,920 non-embedding)
activation    : SwiGLU, hidden_dim 2048 (= 8/3 x 768)
config        : ctx 512 | emb 768 | layers 12 | heads 12 | dropout 0.0
precision     : autocast float16 + GradScaler
batching      : micro 16 x ctx 512 x accum 8 = 65,536 tokens/optimizer step
optimizer     : AdamW lr 0.0003 betas (0.9, 0.95) wd 0.1 clip 1.0
schedule      : 3,572 steps, warmup 178, cosine to 3e-05
train data :  119.84M tokens |    234,067 windows |   14,629 micro-batches | 119.84M tokens/epoch
val   data :   65.54K tokens |        128 windows |        8 micro-batches | 65.54K tokens/epoch
per epoch     : 1,786 optimizer steps
planned total : 3,572 optimizer steps = 234.09M tokens
free VRAM     : 15.4 GB
```

Then it runs:

```
step      0/3,572 | loss 10.8250 | lr 1.68e-06 | 0 tok/s | 0.00 tokens seen | 0.00 h
step     20/3,572 | loss  8.9127 | lr 3.39e-05 | 6,412 tok/s | 1.31M seen | 0.00 h
step     40/3,572 | loss  7.1044 | lr 6.74e-05 | 6,588 tok/s | 2.62M seen | 0.01 h
  [eval] step 0 | train 10.8250 | val 10.8250 | ppl 50257.00 | gap +0.0000  <- best
```

**Sanity numbers.** A random 124M model scores `ln(50257) = 10.825`. If you are
not comfortably below that within 200 steps, something is broken — almost always
the fp16/GradScaler setup.

**Throughput.** Expect **5,000-8,000 tok/s**, i.e. 8-13 s per optimizer step and
about **8-13 hours for 2 epochs**. A published single-T4 measurement for a
nearby configuration gives 7,803 tok/s but with a heavily modified
(custom-kernel) codebase, so plan against the low end.

### If it runs out of memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

Drop the micro-batch and raise accumulation to hold the step size constant:

```bash
python -m llm_from_scratch.pretraining.train --micro-batch 8 --accum-steps 16 --resume
```

The output head is the memory hog, not the attention: `16 x 512 x 50257` logits
is 824 MB in fp16, and `cross_entropy` upcasts it to fp32. If you have room,
*raising* `--micro-batch` is the cheapest throughput win available.

---

## 3. Survive a preemption

This is the part that decides whether the project gets finished.

```bash
# after a disconnect, same command
python -m llm_from_scratch.pretraining.train --resume
```

- Checkpoints every 250 optimizer steps to `/content/checkpoints/latest.pt`
  (~1.5 GB: weights plus AdamW's two moments).
- `best.pt` is written whenever validation loss improves.
- Weights only are mirrored to `--drive-dir` on every checkpoint. The full
  1.5 GB write stays on local disk; Drive's FUSE mount is too slow for the hot
  path.
- Each write goes to a `.partial` file and is then `os.replace`d, so a
  disconnect mid-write cannot corrupt the last good checkpoint.
- RNG state, optimizer state, `GradScaler` state, step count and tokens-seen are
  all saved. Without those a resumed run is not reproducible.
- Every `print` is teed to `train_log.jsonl`'s sibling text log, and structured
  metrics are appended to `train_log.jsonl` as JSON lines.

If the notebook dies *before* the first checkpoint, nothing is lost but the
setup time. Copy the data from Drive and rerun.

**Keep the browser tab open and awake.** Colab kills idle VMs; a training loop
is activity, so you are usually fine, but a throttled background tab can still
trigger it. If your laptop sleeps, the run dies.

---

## 4. Evaluate

### Loss curves

```bash
python -m llm_from_scratch.pretraining.loss_plots \
    --log train_log.jsonl --out /content/loss_curves.png
```

This works after a preemption, because it reads the log rather than the
in-memory history.

### What to expect

| stage | val loss | note |
|---|---|---|
| random init | 10.825 | `ln(50257)`, the uniform floor |
| ~1 epoch | under ~4.2 | if it is above this, suspect LR or the data |
| 2 epochs | ~3.3-3.7 | engineering estimate, not a published figure |

The WikiText-103 validation split is only ~1.16 MB (~245k tokens), so the
periodic eval scores a fixed 65,536-token prefix of it. The number will wobble
by about +/-0.02 between evals; do not over-read a single point.

Published WT-103 perplexities you may find while searching (GPT-2 117M
fine-tuned: 37.50) are **word-normalised**. This is a BPE cross-entropy. They are
not comparable, and it is worth saying so in the README before a reviewer points
it out.

### Honest limitation testing

Run a handful of factual prompts and record what happens. Expect
plausible-sounding and often wrong:

```bash
sarthllm "The capital of France is" --checkpoint /content/checkpoints/best.pt
sarthllm "The capital of France is" --checkpoint /content/checkpoints/best.pt --seed 7
sarthllm "In 1789, the French Revolution" --checkpoint /content/checkpoints/best.pt
sarthllm "The largest planet in the solar system is" --checkpoint /content/checkpoints/best.pt
sarthllm "Water boils at" --checkpoint /content/checkpoints/best.pt --greedy
```

Put the real outputs in the README's limitations section. A documented failure
mode reads as rigour; silence reads as not having checked.

---

## 5. Troubleshooting

**`CUDA out of memory`** — lower `--micro-batch`, raise `--accum-steps` by the
same factor.

**Loss stuck near 10.825** — fp16 without the `GradScaler`. Confirm the banner
says `autocast float16 + GradScaler`.

**Loss goes to `nan`** — the scaler hit its growth limit, usually from an
exploding gradient. Drop `--lr` to 1e-4 and restart from the last checkpoint.

**`RuntimeError: attempted relative import beyond top-level package`** — run
modules as `python -m llm_from_scratch.<module>` from the repo root, never as
`python <file>`.

**`FileNotFoundError: ... not found. Build it first`** — the `.bin` files are
not where `--data-dir` says. Check `meta.json` exists alongside them.

**Throughput far below 5,000 tok/s** — check `nvidia-smi` is not showing
contention, and leave `--compile` off. `torch.compile` has no verified
T4/`sm_75` benefit and its graph-recompile behaviour is a bad trade against a
session that can be preempted at any moment.

---

## Command reference

```bash
# data
python -m llm_from_scratch.data.wikitext_prepare --out-dir DIR [--shard 0.02]
python -m llm_from_scratch.data.wikitext_dataloader --data-dir DIR

# training
python -m llm_from_scratch.pretraining.train --data-dir DIR --resume
    [--epochs 3] [--max-steps N] [--micro-batch 16] [--accum-steps 8]
    [--block-size 512] [--lr 3e-4] [--dropout 0.0] [--precision auto]
    [--ckpt-interval 250] [--eval-interval 250] [--ckpt-dir DIR] [--drive-dir DIR]
    [--compile] [--seed 123]

# plots and samples
python -m llm_from_scratch.pretraining.loss_plots --log train_log.jsonl
sarthllm "prompt" --checkpoint DIR/best.pt [--temperature 0.8] [--top-k 40]
    [--max-tokens 80] [--greedy] [--seed 123] [--repeat 5]
```
