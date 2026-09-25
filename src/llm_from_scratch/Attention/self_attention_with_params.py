import torch
import torch.nn as nn

from .._paths import LLM_BOOK_PATH
from ..data.dataset_dataloader import dataLoader
from ..data.embedding_layer import (BATCH_SIZE, EMBED_DIM, MAX_LENGTH, SHUFFLE,
                                    STRIDE, get_input_embeddings)


def get_input_sequence():
    raw_text = LLM_BOOK_PATH.read_text(encoding="utf-8")
    loader = dataLoader(raw_text, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
                        stride=STRIDE, shuffle=SHUFFLE, drop_last=True)
    input_ids, target_ids = next(iter(loader))

    embeddings = get_input_embeddings(input_ids)
    return embeddings[0]


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)

    inputs = get_input_sequence()
    d_in = inputs.shape[1]
    d_out = 2

    sa_v1 = SelfAttention_v1(d_in, d_out)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print("v1 context vectors:\n", sa_v1(inputs))
    print("\nv2 context vectors:\n", sa_v2(inputs))