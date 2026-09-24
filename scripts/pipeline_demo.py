"""
End-to-end pipeline walkthrough (input stage only):

    raw text -> tiktoken tokenizer -> dataloader -> token embedding
              -> positional embedding -> final input context embedding  (-> logits)

At every stage we print the OUTPUT and the HYPERPARAMETERS that produced it, so you
can see exactly what flows into the next phase (softmax / attention mechanism).

Run:
    uv run python scripts/pipeline_demo.py
"""

import warnings

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch

from llm_from_scratch._paths import LLM_BOOK_PATH
from llm_from_scratch.data.tokenizer import get_tokenizer
from llm_from_scratch.data.dataset_dataloader import dataLoader
from llm_from_scratch.data.embedding_layer import (
    EMBED_DIM,
    MAX_LENGTH,
    STRIDE,
    BATCH_SIZE,
    SHUFFLE,
    VOCAB_SIZE,
    token_embedding_layer,
    pos_embedding_layer,
    get_input_embeddings,
    get_logits,
)

SAMPLES_TO_SHOW = 16
VEC_HEAD = 10

def vec_str(v, head=VEC_HEAD):
    vals = [f"{x:.4f}" for x in v.tolist()[:head]]
    return "[" + ", ".join(vals) + f", ... {len(v) - head} more]"

# ---------------------------------------------------------------------------
# STAGE 1 — Tiktoken tokenizer (GPT-2 BPE)
# ---------------------------------------------------------------------------
raw_text = LLM_BOOK_PATH.read_text(encoding="utf-8")

encoder_name = "gpt2"
enc = get_tokenizer()

print("=" * 72)
print("STAGE 1 | TOKENIZATION (tiktoken)")
print("=" * 72)
print("\n[output] what we start with — raw text:")
print(f"  total characters      : {len(raw_text):,}")
print(f"  first 99 chars        : {raw_text[:99]!r}")
print("\n[hyperparameters]")
print(f"  encoder_name          : {encoder_name}")
print(f"  vocab_size            : {enc.n_vocab:,}")
print(f"  special_tokens_allowed: {{'<|endoftext|>'}}")
print("\n[output] after encode():")

token_ids = enc.encode(raw_text, allowed_special={"<|endoftext|>"})
print(f"  total tokens          : {len(token_ids):,}")
print(f"  first {SAMPLES_TO_SHOW} token ids : {token_ids[:SAMPLES_TO_SHOW]}")
decoded = enc.decode([token_ids[i] for i in range(SAMPLES_TO_SHOW)])
print(f"  first {SAMPLES_TO_SHOW} decoded       : {decoded!r}")

# ---------------------------------------------------------------------------
# STAGE 2 — Dataloader (input/target windows, shifted by 1)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STAGE 2 | DATALOADER")
print("=" * 72)
print("\n[hyperparameters]")
print(f"  batch_size            : {BATCH_SIZE}")
print(f"  context_length        : {MAX_LENGTH}")
print(f"  stride                : {STRIDE}")
print(f"  shuffle               : {SHUFFLE}")
print(f"  drop_last             : True")
print(f"  num_workers           : 0")

loader = dataLoader(
    raw_text,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    stride=STRIDE,
    shuffle=SHUFFLE,
    drop_last=True,
)
dataset = loader.dataset
print(f"\n  total samples (windows): {len(dataset):,}")

input_ids, target_ids = next(iter(loader))  # first batch (shuffled)
print("\n[output] one batch from the dataloader:")
print(f"  input_ids  shape: {tuple(input_ids.shape)}  dtype: {input_ids.dtype}")
print(f"  target_ids shape: {tuple(target_ids.shape)}  dtype: {target_ids.dtype}")
print(f"  input_ids  row 0: {input_ids[0].tolist()}")
print(f"  target_ids row 0: {target_ids[0].tolist()}")
print(f"  target row 0 = input row 0 shifted left by 1 (next-token prediction): "
      f"{input_ids[0][0].item()} -> {target_ids[0][0].item()}")

# ---------------------------------------------------------------------------
# STAGE 3 — Token embedding
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STAGE 3 | TOKEN EMBEDDING")
print("=" * 72)
print("\n[hyperparameters]")
print(f"  vocab_size            : {VOCAB_SIZE:,}")
print(f"  embed_dim             : {EMBED_DIM}")
print(f"  weight_tying          : output_layer shares token_embedding_layer.weight")

token_embeddings = token_embedding_layer(input_ids)
print("\n[output]")
print(f"  token_embeddings shape: {tuple(token_embeddings.shape)}")
print(f"  per-token vector      : {vec_str(token_embeddings[0, 0])}")

# ---------------------------------------------------------------------------
# STAGE 4 — Positional embedding (learned absolute positions, added)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STAGE 4 | POSITIONAL EMBEDDING (learned, additive)")
print("=" * 72)
print("\n[hyperparameters]")
print(f"  num_embeddings        : {MAX_LENGTH}  (one slot per context position)")
print(f"  embed_dim             : {EMBED_DIM}")
print(f"  scheme                : learned absolute positions, ADDED to token vectors")

seq_len = input_ids.shape[1]
pos_ids = torch.arange(seq_len)
pos_embeddings = pos_embedding_layer(pos_ids)
print("\n[output]")
print(f"  pos_ids        shape  : {tuple(pos_ids.shape)}")
print(f"  pos_embeddings shape  : {tuple(pos_embeddings.shape)}")
print(f"  pos_ids               : {pos_ids.tolist()[:VEC_HEAD]} ...")
print(f"  embedding at pos 0    : {vec_str(pos_embeddings[0])}")
print(f"  embedding at pos 1    : {vec_str(pos_embeddings[1])}")

# ---------------------------------------------------------------------------
# STAGE 5 — Final input context embedding (ready for attention)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STAGE 5 | FINAL INPUT CONTEXT EMBEDDING -> next phase: ATTENTION")
print("=" * 72)
print("\n[formula]")
print("  input_embedding = token_embedding + pos_embedding   (broadcast pos over batch)")

input_embeddings = get_input_embeddings(input_ids)
print("\n[output]")
print(f"  input_embeddings shape: {tuple(input_embeddings.shape)}")
print(f"  determinant check     : token+pos == get_input_embeddings -> "
      f"{torch.equal(token_embeddings + pos_embeddings.unsqueeze(0), input_embeddings)}")
print(f"  row 0 pos 0 vector    : {vec_str(input_embeddings[0, 0])}")
print(f"  mean | std | min | max: "
      f"{input_embeddings.mean():.3f} | {input_embeddings.std():.3f} | "
      f"{input_embeddings.min():.3f} | {input_embeddings.max():.3f}")

print("\n  >>> This (batch_size, context_length, embed_dim) tensor of shape "
      f"{tuple(input_embeddings.shape)} is the INPUT embedding for the next phase: "
      "the attention mechanism.")

print("\n[bonus] one more step through the output head (weight-tied):")
logits = get_logits(input_embeddings)
print(f"  logits shape: {tuple(logits.shape)}  (batch, context_length, vocab)")

print("\nDone.")