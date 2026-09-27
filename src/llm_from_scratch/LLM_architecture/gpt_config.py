"""Model config for the WikiText-103 pretraining run.

Sized for a single Google Colab T4 (16 GB VRAM, sm_75).

    vocab_size     50257   GPT-2 BPE vocabulary. Do not round to 50304: that
                          padding trick exists to make tensor-core matmuls
                          faster, and the T4's kernels are not sensitive to it.
    context_length  512    Half of GPT-2 small's 1024. At d=768 attention is
                          ~10% of the FLOPs at 512 vs ~18% at 1024, and the
                          shorter context halves activation memory, which is
                          what lets micro-batch 16 fit at all.
    emb_dim        768     12 heads x 64 dims. Unchanged from GPT-2 small.
    n_heads         12
    n_layers        12
    ffn_hidden_dim 2048    = swiglu_hidden_dim(768). Parameter-matches the
                          original 4*d = 3072 GELU FFN exactly.
    drop_rate       0.0    Was 0.1 in the book config. Dropout fights
                          overfitting, and this run is nowhere near
                          overfitting: 124M parameters against ~120M training
                          tokens is ~1 token/param, six times short of
                          Chinchilla's 20. Regularising an under-trained model
                          only slows convergence. Revisit if val loss starts
                          climbing while train loss keeps falling. The exact
                          token count is whatever wikitext_prepare.py reports.
    qkv_bias      True     Matches published GPT-2 small.
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "ffn_hidden_dim": 2048,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
