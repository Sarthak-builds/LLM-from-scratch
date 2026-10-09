# sarthLLM

A GPT built from scratch. "From scratch" means the weights, not just the
layers — tokenizer, embeddings, causal multi-head attention, LayerNorm,
SwiGLU, and (eventually) the pretraining loop are all in this repo. No
pretrained weights are downloaded or loaded at any point.

Follows *Build a Large Language Model (From Scratch)* by Sebastian Raschka —
the book text ships in `assets/llm-book.txt` — with a few deliberate detours
(SwiGLU instead of GELU, WikiText-103 instead of the book's tiny corpus).

## Where we are

The input half of the pipeline is built and runs end to end. The model half
is what's being written next.

| stage | status |
|---|---|
| BPE tokenizer (tiktoken `gpt2`) + hand-rolled regex BPE experiment | done |
| sliding-window dataloader over raw text | done |
| WikiText-103 -> uint16 `.bin` + memmap windowed loader | done |
| token + positional embeddings, logits tied to token weights | done |
| attention: simple -> learnable -> causal masked -> multi-head | done |
| LayerNorm, SwiGLU | done |
| sampling: greedy + top-k/temperature | done |
| `GPT_CONFIG_124M` | done (the target, nothing trains on it yet) |
| FFN + transformer block + `GPT` class | next |
| pretraining loop, loss curves, checkpoints | next |
| `sarthllm` CLI | next |

It will eventually **continue text**. It will not chat, follow instructions,
or answer questions reliably — that is a deliberate scope decision, documented
in [Limitations](#limitations) rather than hidden.

## Quick start

Everything below needs nothing but CPU and an internet connection for step 1.
When the model and training loop land, they'll be added here — the plan is
documented in [`COLAB.md`](COLAB.md).

```bash
# 1. fetch + tokenize WikiText-103 (~190 MB download, writes uint16 .bin files)
python -m llm_from_scratch.data.wikitext_prepare --out-dir data/wikitext

# 2. inspect the memmap windowed loaders (CPU only)
python -m llm_from_scratch.data.wikitext_dataloader --data-dir data/wikitext

# 3. see the whole input pipeline print its stages: raw text -> tiktoken ->
#    dataloader -> token + positional embeddings -> input context embedding
uv run python scripts/pipeline_demo.py
```

A 2-minute smoke test before downloading the whole 103M-word corpus:

```bash
python -m llm_from_scratch.data.wikitext_prepare --out-dir data/wikitext --shard 0.02
```

## Architecture

This is the **target**: `GPT_CONFIG_124M` in
`src/llm_from_scratch/LLM_architecture/gpt_config.py`. Of the pieces in the
table, LayerNorm, SwiGLU and causal multi-head attention exist today; the
transformer block that wires them together and the `GPT` class that stacks
them do not yet.

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

The attention modules are built at a small demo scale on the way up (an
unstructured demo on the book text first, then at GPT-2 size).

The SwiGLU hidden width is `8/3 x 768 = 2048` rather than the usual
`4 x 768 = 3072`, because a gated feed-forward needs three projection matrices
where the GELU version needed two. The arithmetic works out exactly:

```
GELU   : 2 x 768 x 3072 = 4,718,592
SwiGLU : 3 x 768 x 2048 = 4,718,592    <- identical
```

So the feed-forward costs the same as in GPT-2 small, and the model stays at
124M. `src/llm_from_scratch/LLM_architecture/swiglu.py` has the derivation.

## Training setup (planned)

None of this is running yet — the model and the loop don't exist. This is the
recipe the pretraining code is being written against, sized for a single
Google Colab T4 (16 GB VRAM):

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
backward pass, so the loss must be rescaled. The rationale for every number is
sketched in `GPT_CONFIG_124M`'s docstring.

## Layout

```
src/llm_from_scratch/
  Attention/           self-attention built up: no params -> one head -> causal -> multi-head
  data/                tokenizer, sliding-window dataset, embeddings, WikiText prepare + memmap loader
  tokenization/        standalone BPE experiments (regex, tiktoken)
  LLM_architecture/    config, LayerNorm, SwiGLU, text-generation sampling
  _paths.py            repo/assets paths (project root, the book text)
scripts/
  pipeline_demo.py     prints every stage of the input pipeline end to end
```

`Attention/` is a progression, not a pile of alternatives: each file adds one
idea to the one before it, and only `multi_head_attention.py` feeds into the
model.

## Study notes

`documentation/` holds per-module notes explaining *why* each piece works.
It is gitignored — those are working notes, not part of the package.

## Limitations

Being honest about where this actually is:

- **There is no trained model.** The FFN, transformer block and `GPT` class
  aren't written, so nothing generates text yet. The loss, perplexity and
  sample-output sections elsewhere in these docs describe the intended
  end state, not measured results.
- **Two scales are in the repo on purpose.** The `data/` embeddings demo runs
  at `EMBED_DIM=256, MAX_LENGTH=256` on the book text; the attention and
  `LLM_architecture` modules are the GPT-2-small-sized building blocks
  (768/512/12). They aren't wired together yet, so don't read the demo's
  numbers as the model's.
- **Once trained, it will produce fluent nonsense.** Wikipedia prose is
  predictable enough that surface grammar and topic drift look right. That is
  not the same as being correct. It is a next-token predictor with no
  instruction tuning and no chat template.
- **Under-trained by design.** 124M parameters against roughly 130M training
  tokens is about **1 token per parameter**. Chinchilla's rule of thumb is 20.
  It is compute-limited by what fits in a 12-hour Colab session.
- **Not Chinchilla-comparable.** Published WikiText-103 perplexities are
  word-normalised; this is a BPE cross-entropy. Different quantities, not
  comparable numbers.

## Next up

1. Feed-forward + a pre-norm transformer block that composes attention,
   LayerNorm and SwiGLU.
2. A `GPT` class that stacks the block and ties the output head to the token
   embedding.
3. The pretraining loop: LR warmup/cosine, fp16 `GradScaler`, checkpointing,
   resumable runs.
4. Loss/perplexity curves and the `sarthllm` CLI.

## Credits

The architecture and the planned training recipe follow *Build a Large
Language Model (From Scratch)* by Sebastian Raschka, whose text is in
`assets/llm-book.txt`. The SwiGLU feed-forward follows Shazeer, *GLU Variants
Improve Transformer* (arXiv:2002.05202), as used in LLaMA. The (planned)
optimizer hyperparameters and warmup/cosine schedule follow nanoGPT. The
corpus is [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext)
(CC-BY-SA-4.0). The GPT-2 BPE tokenizer is OpenAI's `tiktoken` release — a
tokenizer only, no model weights are used.
