import torch
import torch.nn as nn

from ..data.embedding_layer import EMBED_DIM
from .self_attention_with_params import get_input_sequence


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)

    inputs = get_input_sequence()               # (seq_len, embed_dim)
    inputs = inputs.unsqueeze(0)                # (1, seq_len, embed_dim)
    context_length = inputs.shape[1]

    causal_attention = CausalAttention(
        d_in=EMBED_DIM,
        d_out=EMBED_DIM,
        context_length=context_length,
        dropout=0.5,
    )

    context_vec = causal_attention(inputs)
    print("Input shape:            ", tuple(inputs.shape))
    print("Context vector shape:   ", tuple(context_vec.shape))
    print("First 5 context vectors:\n", context_vec[0, :5])