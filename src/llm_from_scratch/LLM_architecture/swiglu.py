"""SwiGLU — the gated feed-forward activation used by LLaMA, Mistral and Qwen.

Replaces the plain GELU feed-forward of the original GPT-2 block. The idea
(Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202) is to give the
feed-forward network a *gate*: one projection decides how much of another
projection may pass.

    GELU FFN:    h = GELU(x @ W_up)          @ W_down       2 matrices
    SwiGLU FFN:  h = SiLU(x @ W_gate) * (x @ W_up) @ W_down 3 matrices

`SiLU` (a.k.a. swish) is `x * sigmoid(x)`, available as `F.silu`.

The extra matrix is not free, so the hidden width is shrunk from the usual
`4 * emb_dim` to `8/3 * emb_dim`, which makes the parameter count match the
GELU version almost exactly. See `swiglu_hidden_dim`.
"""

import torch.nn as nn
import torch.nn.functional as F

# Why a hidden width of 8/3 and not 4? A GELU FFN spends 2 * d * 4d parameters.
# A gated FFN needs 3 matrices, so to spend the same budget the hidden width h
# must satisfy 3 * d * h == 2 * d * 4d, i.e. h == 8/3 * d.
FFN_HIDDEN_RATIO = 8 / 3
# Rounding up to a multiple of a power of two keeps the matmul shapes friendly.
# The LLaMA reference implementation uses 256; 64 already gets the tensor cores
# to line up and leaves d=768 at exactly 2048.
FFN_HIDDEN_MULTIPLE_OF = 64


def swiglu_hidden_dim(emb_dim, multiple_of=FFN_HIDDEN_MULTIPLE_OF):
    """Hidden width for a SwiGLU FFN that is parameter-matched to a 4*d GELU one.

    h0 = round(2 * (4 * emb_dim) / 3)  ==  round(8/3 * emb_dim)
    h  = smallest multiple of `multiple_of` that is >= h0

    >>> swiglu_hidden_dim(768)
    2048
    >>> swiglu_hidden_dim(1024)
    2752
    >>> swiglu_hidden_dim(4096)
    10944

    LLaMA-7B's published 11008 comes from the same rule at a coarser multiple:
    `swiglu_hidden_dim(4096, multiple_of=256) == 11008`. The default 64 is
    tighter, so it lands lower and wastes fewer parameters on padding.
    """
    hidden = int(2 * (4 * emb_dim) / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """The activation half of the block: SiLU-gate times a linear branch.

    Takes the two projections separately rather than one fused `2h` matrix so
    the gate and the content branch stay readable. Fusing them into a single
    `nn.Linear(d, 2h)` is a pure speed optimisation with identical maths.
    """

    def forward(self, gate, up):
        return F.silu(gate) * up
