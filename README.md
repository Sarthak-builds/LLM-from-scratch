# sarthLLM

A GPT built from scratch — tokenizer, embeddings, causal multi-head attention,
transformer blocks, SwiGLU feed-forward, AdamW, a training loop, and a CLI —
then pretrained on **WikiText-103** and nothing else. No pretrained weights are
downloaded or loaded at any point; every parameter here is learned from
scratch on the corpus you prepare.

124,000,512 parameters. ~124M.

## What it does, and what it does not

It **continues text**. That is the whole interface.

```console
$ sarthllm "The capital of France is"
  the capital of France is a city that has been the seat of government ...
```

It does **not** chat, follow instructions, or answer questions reliably. See
[Limitations](#limitations) — this is a deliberate scope decision, documented
rather than hidden.

## Quick start

Training runs on a GPU. It was developed against a single Google Colab T4
(16 GB), and [`COLAB.md`](COLAB.md) is the full walkthrough including how to
survive session preemption.

```bash
# 1. fetch + tokenize WikiText-103 (~190 MB download, writes a ~229 MB .bin)
python -m llm_from_scratch.data.wikitext_prepare --out-dir data/wikitext

# 2. inspect the loaders without touching a GPU
python -m llm_from_scratch.data.wikitext_dataloader --data-dir data/wikitext

# 3. train (~12 h for 2 epochs on a T4; resumes from checkpoints/latest.pt)
python -m llm_from_scratch.pretraining.train

# 4. plot the curves
python -m llm_from_scratch.pretraining.loss_plots --log train_log.jsonl

# 5. sample
sarthllm "The capital of France is" --checkpoint checkpoints/best.pt
```

A 5-minute smoke test before committing GPU hours:

```bash
python -m llm_from_scratch.data.wikitext_prepare --out-dir data/wikitext --shard 0.02
python -m llm_from_scratch.pretraining.train --data-dir data/wikitext \
    --max-steps 50 --eval-interval 25 --log-interval 5
```

## Architecture

| | |
|---|---|
| layers | 12 |
| heads | 12 (64 dims each) |
| embedding dim | 768 |
| context length | 512 |
| vocabulary | 50257 (GPT-2 BPE) |
| feed-forward | **SwiGLU**, hidden 2048 |
| normalisation | LayerNorm, pre-norm |
| weight tying | output head shares the token embedding |
| dropout | 0.0 |

The SwiGLU hidden width is `8/3 x 768 = 2048` rather than the usual
`4 x 768 = 3072`, because a gated feed-forward needs three projection matrices
where the GELU version needed two. The arithmetic works out exactly:

```
GELU   : 2 x 768 x 3072 = 4,718,592
SwiGLU : 3 x 768 x 2048 = 4,718,592    <- identical
```

So the feed-forward costs the same as in GPT-2 small, and the model stays at
124M. `src/llm_from_scratch/LLM_architecture/swiglu.py` has the derivation.

## Training setup

| | |
|---|---|
| precision | fp32 weights, **fp16** autocast, `GradScaler` |
| micro-batch x accum | 16 x 512 x 8 = 65,536 tokens/optimizer step |
| optimizer | AdamW, betas (0.9, 0.95), wd 0.1 on 2-D weights only |
| LR | 3e-4 peak, 5% warmup, cosine to 3e-5 |
| gradient clip | 1.0 |
| epochs | 2 (~260M tokens) |

The precision choice is forced by the hardware, not a preference: the T4 is
`sm_75`, and bf16 tensor cores only exist from `sm_80`. fp16 underflows in the
backward pass, so the loss must be rescaled. See the module docstring in
`src/llm_from_scratch/pretraining/train.py`.

## Layout

```
src/llm_from_scratch/
  Attention/           self-attention built up: no params -> one head -> causal -> multi-head
  data/                tokenization, sliding-window dataset, WikiText prepare + dataloader
  tokenization/        standalone BPE experiments (regex, tiktoken)
  LLM_architecture/    GELU-free stack: config, LayerNorm, SwiGLU, FFN, block, GPT, generation
  pretraining/         LR schedule, training loop, loss/perplexity, evaluation, plots
  cli.py               the sarthllm command
```

`Attention/` is a progression, not a pile of alternatives: each file adds one
idea to the one before it, and only `multi_head_attention.py` is used by the
trained model.

## Study notes

`documentation/` holds per-module notes explaining *why* each piece works.
It is gitignored — those are working notes, not part of the package.

## Limitations

Stated up front, because the failure modes are predictable and worth writing
down.

- **Under-trained by design.** 124M parameters against roughly 130M training
  tokens is about **1 token per parameter**. Chinchilla's rule of thumb is 20.
  The model is compute-limited, and this run is what fits in a 12-hour
  Colab session, not what the architecture could absorb with 2.4B tokens.
- **It will produce fluent nonsense.** Wikipedia prose is predictable enough
  that surface grammar, syntax and topic drift all look right. That is not the
  same as being correct. Ask it a factual question and it will sometimes
  answer with confident, well-formed, wrong text.
- **No instruction tuning.** There is no chat template, no system prompt, no
  turn structure. It is a next-token predictor and nothing more.
- **512-token memory.** Prompts longer than 512 tokens raise rather than being
  silently truncated.
- **Not Chinchilla-comparable.** Published WikiText-103 perplexities are
  word-normalised; this is a BPE cross-entropy. The two are different
  quantities and the numbers are not comparable.

## Credits

The architecture and training recipe follow *Build a Large Language Model (From
Scratch)* by Sebastian Raschka, whose text is in `assets/llm-book.txt`. The
SwiGLU feed-forward follows Shazeer, *GLU Variants Improve Transformer*
(arXiv:2002.05202), as used in LLaMA. The optimizer hyperparameters and
warmup/cosine schedule follow nanoGPT. The corpus is
[WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (CC-BY-SA-4.0).
The GPT-2 BPE tokenizer is OpenAI's `tiktoken` release — a tokenizer only, no
model weights are used.
