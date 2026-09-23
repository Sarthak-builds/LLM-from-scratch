import torch.nn as nn
from .._paths import LLM_BOOK_PATH
from .tokenizer import get_tokenizer
from .dataset_dataloader import dataLoader

EMBED_DIM  = 256
MAX_LENGTH = 256
STRIDE     = 128
BATCH_SIZE = 4
SHUFFLE    = True

tokenizer  = get_tokenizer()
VOCAB_SIZE = tokenizer.n_vocab

embedding_layer = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBED_DIM)


if __name__ == "__main__":
    raw_text = LLM_BOOK_PATH.read_text(encoding="utf-8")

    loader = dataLoader(raw_text, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
                         stride=STRIDE, shuffle=SHUFFLE, drop_last=True)
    input_ids, target_ids = next(iter(loader))

    token_embeddings = embedding_layer(input_ids)
    print(f"Embedding output shape: {token_embeddings.shape}")
    print(f"  -> (batch_size={BATCH_SIZE}, seq_len={MAX_LENGTH}, embed_dim={EMBED_DIM})")