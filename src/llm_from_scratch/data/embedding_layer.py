import torch
import torch.nn as nn
from .._paths import LLM_BOOK_PATH
from .tokenizer import get_tokenizer
from .dataset_dataloader import dataLoader

EMBED_DIM  = 256
MAX_LENGTH = 256   # must match context_length used everywhere — positional table is sized to this
STRIDE     = 128
BATCH_SIZE = 4
SHUFFLE    = True

tokenizer  = get_tokenizer()
VOCAB_SIZE = tokenizer.n_vocab

token_embedding_layer = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBED_DIM)
pos_embedding_layer   = nn.Embedding(num_embeddings=MAX_LENGTH, embedding_dim=EMBED_DIM)


def get_input_embeddings(input_ids):
    """
    input_ids: (batch_size, seq_len) token IDs
    returns:   (batch_size, seq_len, embed_dim) — token + positional, combined
    """
    token_embeddings = token_embedding_layer(input_ids)           # (batch, seq_len, embed_dim)

    seq_len = input_ids.shape[1]
    pos_ids = torch.arange(seq_len, device=input_ids.device)      # (seq_len,)
    pos_embeddings = pos_embedding_layer(pos_ids)                 # (seq_len, embed_dim)

    return token_embeddings + pos_embeddings                      # broadcasts pos across batch dim


if __name__ == "__main__":
    raw_text = LLM_BOOK_PATH.read_text(encoding="utf-8")

    loader = dataLoader(raw_text, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
                         stride=STRIDE, shuffle=SHUFFLE, drop_last=True)
    input_ids, target_ids = next(iter(loader))

    input_embeddings = get_input_embeddings(input_ids)

    print(f"Token IDs shape:        {input_ids.shape}")
    print(f"Input embeddings shape: {input_embeddings.shape}")