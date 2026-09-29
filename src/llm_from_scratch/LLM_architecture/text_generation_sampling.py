"""Text generation: turning model logits into text.

Two strategies, both operating on a token-id tensor and returning a token-id
tensor (prompt included):

    generate_text_simple  greedy — argmax at every step. Deterministic, and it
                          will happily loop forever on a small model.
    generate_text_sample  nucleus-ish sampling — top-k filter, then temperature,
                          then multinomial draw. Seed the `generator` to make it
                          reproducible.

Both are autoregressive: feed the whole context, take the logits at the last
position, append the chosen token, repeat. The context is re-cropped to the
last `context_size` tokens every step, which is why generation is O(n^2) in
compute for n new tokens.
"""

import torch
import torch.nn.functional as F

TEMPERATURE = 1.0
TOP_K = None


def apply_temperature(logits, temperature=TEMPERATURE):
    """Divide the logits by the temperature before the softmax.

    - `temperature < 1` sharpens the distribution -> closer to greedy, more
      predictable text.
    - `temperature > 1` flattens it -> every token is more likely, more random
      (and eventually just noise).
    - `temperature == 1` leaves the model's own distribution alone.

    >>> apply_temperature(torch.tensor([2.0, 4.0]), 1.0).tolist()
    [2.0, 4.0]
    >>> apply_temperature(torch.tensor([2.0, 4.0]), 2.0).tolist()   # halved
    [1.0, 2.0]
    >>> float(apply_temperature(torch.ones(4), 0.5).max())         # < 1 sharpens
    2.0
    >>> apply_temperature(torch.ones(4), 0.0)
    Traceback (most recent call last):
        ...
    ValueError: temperature must be > 0
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def apply_top_k(logits, top_k=TOP_K):
    """Keep only the `top_k` highest logits per position and mask the rest with
    -inf, so the tail of the vocabulary can never be sampled.

    `top_k=None` (or 0) disables filtering. This is what stops a trained model
    from occasionally emitting a token that has no business being there.

    Works for logits of any rank: the last dimension is the vocabulary, and
    `[..., -1:]` keeps a trailing singleton so the threshold broadcasts against
    both `(B, V)` and `(B, T, V)`. (Taking `[..., -1, :]` instead works only for
    the 3-D case and silently mis-slices the 2-D one.)

    >>> lg = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
    >>> apply_top_k(lg, 2).tolist()          # top 2 of [1,5,2,4,3] are 5 and 4
    [[-inf, 5.0, -inf, 4.0, -inf]]
    >>> apply_top_k(lg, 2).gt(float("-inf")).sum().item()   # exactly k survive
    2
    >>> bool(torch.equal(apply_top_k(lg, 0), lg))          # 0 disables
    True
    >>> apply_top_k(lg, 1).tolist()                       # k=1 keeps argmax
    [[-inf, 5.0, -inf, -inf, -inf]]

    Generation passes 2-D `(B, V)` logits and a full forward pass produces 3-D
    `(B, T, V)`, so both have to work:

    >>> for shape in ((2, 9), (2, 3, 9)):
    ...     out = apply_top_k(torch.randn(*shape), 4)
    ...     assert out.shape == shape
    ...     assert torch.equal(out.gt(float("-inf")).sum(-1),
    ...                        torch.full(shape[:-1], 4))
    """
    if top_k is None or top_k == 0:
        return logits
    if not 0 < top_k <= logits.size(-1):
        raise ValueError(
            f"top_k must be between 1 and {logits.size(-1)}, got {top_k}"
        )
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy decoding: always take the most probable next token."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = F.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text_sample(model, idx, max_new_tokens, context_size,
                         temperature=TEMPERATURE, top_k=TOP_K,
                         generator=None):
    """Sample the next token: top-k filter, then temperature, then multinomial."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        logits = apply_top_k(logits, top_k)
        logits = apply_temperature(logits, temperature)
        probas = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1,
                                     generator=generator)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(model, idx, max_new_tokens, context_size,
             temperature=TEMPERATURE, top_k=TOP_K, generator=None):
    """One entry point: greedy when temperature is None, sampled otherwise."""
    if temperature is None:
        return generate_text_simple(model, idx, max_new_tokens, context_size)
    return generate_text_sample(model, idx, max_new_tokens, context_size,
                                temperature=temperature, top_k=top_k,
                                generator=generator)


if __name__ == "__main__":
    import torch.nn.functional as F

    from ..data.tokenizer import get_tokenizer
    from .GPTmodel import GPTModel
    from .gpt_config import GPT_CONFIG_124M

    torch.manual_seed(123)

    # top-k must work for the 2-D (B, V) logits generation uses and for the
    # 3-D (B, T, V) logits a full forward pass produces.
    for shape in ((4, 50257), (2, 7, 50257)):
        lg = torch.randn(*shape)
        for k in (1, 5, 40):
            out = apply_top_k(lg, k)
            kept = out.gt(float("-inf")).sum(-1)
            assert out.shape == lg.shape, f"shape changed for {shape}"
            assert torch.equal(kept, torch.full_like(kept, k)), \
                f"{shape} k={k}: kept {kept.tolist()} != {k}"
    assert torch.equal(apply_top_k(lg, None), lg), "top_k=None must be a no-op"
    print("top_k OK for 2-D and 3-D logits\n")

    assert abs(apply_temperature(torch.ones(3), 2.0).max().item() - 0.5) < 1e-6
    print("temperature OK\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.eval()
    tokenizer = get_tokenizer()

    start_context = "Hello, I am"
    encoded = torch.tensor(
        tokenizer.encode(start_context, allowed_special={"<|endoftext|>"}),
        dtype=torch.long,
    ).unsqueeze(0).to(device)
    context_size = GPT_CONFIG_124M["context_length"]

    settings = [
        ("greedy (argmax)          ", None, None, None),
        ("sampling T=1.0, top_k=5  ", 1.0, 5, 123),
        ("sampling T=0.8, top_k=5  ", 0.8, 5, 123),
        ("sampling T=1.2, top_k=50 ", 1.2, 50, 123),
    ]

    print("(untrained model — output is noise, this only checks the plumbing)")
    for label, temperature, top_k, seed in settings:
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
        out_ids = generate(model, encoded, max_new_tokens=20,
                           context_size=context_size, temperature=temperature,
                           top_k=top_k, generator=gen)
        assert out_ids.shape[1] == encoded.shape[1] + 20, "wrong length"
        print(f"{label}: {tokenizer.decode(out_ids[0].tolist())!r}")
    print("\nOK: greedy and sampled generation both produce 20 new tokens")
