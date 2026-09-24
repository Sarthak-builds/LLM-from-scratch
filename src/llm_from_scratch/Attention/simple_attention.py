import torch


def simple_self_attention(inputs):

    seq_len = inputs.shape[0]
    attn_scores = inputs @ inputs.T   # (seq_len, seq_len)
    attn_weights = torch.softmax(attn_scores, dim=-1)   
    context_vectors = attn_weights @ inputs   

    return context_vectors, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)
    inputs = torch.rand(6, 3)   # pretend: 6 tokens, embed_dim=3

    context_vectors, attn_weights = simple_self_attention(inputs)

    print("Input embeddings:\n", inputs)
    print("\nAttention weights:\n", attn_weights)
    print("\nContext vectors:\n", context_vectors)
    print("\nEach row of attn_weights sums to:", attn_weights.sum(dim=-1))