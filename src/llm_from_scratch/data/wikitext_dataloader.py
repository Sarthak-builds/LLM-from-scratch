"""Step 2 of training: serve (input, target) windows out of the .bin files.

    python -m llm_from_scratch.data.wikitext_dataloader --data-dir data/wikitext

Differences from the book's `LLMDataset`, and why:

* **Memory-mapped, not materialised.** `LLMDataset` tokenizes a string and
  builds a `list` of every window up front, so its RAM use is
  `O(corpus)`. WikiText-103 is ~120M training tokens; holding even one `int64`
  copy is a gigabyte, and the book-sized sliding window would be ~250k rows.
  Here the file is opened with `np.memmap`, so a 230 MiB file costs a few KB of
  resident memory and the OS pages it in as the sampler walks forward.

* **No window overlap.** `LLMDataset` advances by `stride` and defaults that to
  half the window, which duplicates ~50% of every token's gradient signal.
  Overlap is useful when the corpus is small and you want many distinct
  samples. For pretraining, non-overlapping windows cover the corpus exactly
  once per epoch at half the cost, so `stride == block_size`.

* **Validation is a fixed prefix, never shuffled.** The WikiText validation
  split is only ~1.16 MB of text (~0.25M BPE tokens, a fraction of one percent
  of train), so a full pass is cheap. Capping `val_tokens` keeps the periodic
  eval from dominating the step time, and reading the same prefix every time
  keeps the curve comparable.

A window at offset `s` needs ids `s : s + block + 1` so that
`targets[:, :-1] == inputs[:, 1:]` can hold. Hence `num_windows` is
`(n_tokens - block_size - 1) // stride + 1`.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TOKEN_DTYPE = np.uint16


class _FakeStore:
    """Minimal stand-in for `TokenStore`, for the doctests above."""

    def __init__(self, values):
        self.tokens = np.asarray(list(values), dtype=TOKEN_DTYPE)
        self.n_tokens = self.tokens.size


# Enough windows to make the periodic eval cheap but stable: 128 * 512 = 65,536
# tokens, about a quarter of the whole validation split.
DEFAULT_VAL_TOKENS = 65_536


class TokenStore:
    """A memory-mapped token file, read-only, with its count in `n_tokens`."""

    def __init__(self, path, limit=None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build it first:\n"
                f"  python -m llm_from_scratch.data.wikitext_prepare "
                f"--out-dir {path.parent}"
            )
        self.path = path
        self.tokens = np.memmap(path, dtype=TOKEN_DTYPE, mode="r")

        available = self.tokens.size
        if limit is not None:
            if limit > available:
                raise ValueError(
                    f"requested {limit:,} tokens but {path.name} holds "
                    f"{available:,}"
                )
            # A memmap slice is a view, so this stays memory-mapped.
            self.tokens = self.tokens[:limit]
        self.n_tokens = self.tokens.size

    def __repr__(self):
        return f"TokenStore({self.path.name}, n_tokens={self.n_tokens:,})"


class WindowDataset(Dataset):
    """Contiguous non-overlapping windows of `block_size + 1` tokens.

    `len(dataset)` is the number of windows, so an epoch covers
    `len(dataset) * block_size` tokens.

    >>> ds = WindowDataset(_FakeStore(range(20)), block_size=4)
    >>> len(ds)
    4
    >>> inp, tgt = ds[0]
    >>> inp.tolist(), tgt.tolist()
    ([0, 1, 2, 3], [1, 2, 3, 4])
    >>> torch.equal(tgt[:-1], inp[1:])       # the invariant that matters
    True
    >>> ds.stride == ds.block_size           # no overlap by default
    True
    """

    def __init__(self, store, block_size, stride=None):
        if block_size < 2:
            raise ValueError("block_size must be >= 2 to form (input, target) pairs")
        self.store = store
        self.block_size = block_size
        self.stride = block_size if stride is None else stride

        # +1 for the extra token the shifted targets need.
        span = block_size + 1
        if store.n_tokens < span:
            raise ValueError(
                f"{store.n_tokens:,} tokens is fewer than one {span}-token window"
            )
        self.n_windows = (store.n_tokens - span) // self.stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.store.tokens[start: start + self.block_size + 1]
        ids = torch.from_numpy(chunk.astype(np.int64))
        return ids[:-1], ids[1:]


def get_wikitext_loaders(data_dir, block_size, batch_size, val_tokens=None,
                         val_batch_size=None, num_workers=0, seed=123,
                         pin_memory=True, shuffle_train=True):
    """Build the train and validation loaders from a prepared data dir.

    `val_tokens` caps how much of the validation split the periodic eval reads.
    `val_batch_size` defaults to `batch_size` so the eval sees exactly the same
    batch shape as training — a different shape would change how many windows
    the fixed prefix covers and make the numbers drift between runs.

    `val_tokens` is clamped to what is actually on disk. A `--shard 0.02` run
    leaves only a few thousand validation tokens, well under the 65,536 default,
    and raising here would make the smoke test impossible to run without also
    having to remember to lower the flag.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Build the data first:\n"
            f"  python -m llm_from_scratch.data.wikitext_prepare "
            f"--out-dir {data_dir}"
        )
    meta = json.loads(meta_path.read_text("utf-8"))

    train_store = TokenStore(data_dir / meta["splits"]["train"]["file"])
    val_file = data_dir / meta["splits"]["val"]["file"]
    available = val_file.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
    if val_tokens is not None and val_tokens > available:
        print(f"  note: val_tokens={val_tokens:,} exceeds the {available:,} "
              f"tokens on disk — using all of them")
        val_tokens = available
    val_store = TokenStore(val_file, limit=val_tokens)

    train_ds = WindowDataset(train_store, block_size)
    val_ds = WindowDataset(val_store, block_size)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        drop_last=True, num_workers=num_workers, pin_memory=pin_memory,
        generator=generator, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size or batch_size, shuffle=False,
        drop_last=True, num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader, meta


def loader_stats(loader):
    ds = loader.dataset
    return {
        "tokens": ds.store.n_tokens,
        "windows": len(ds),
        "batches": len(loader),
        "tokens_per_epoch": len(ds) * ds.block_size,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "wikitext")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_VAL_TOKENS)
    args = parser.parse_args()

    train_loader, val_loader, meta = get_wikitext_loaders(
        args.data_dir, args.block_size, args.batch_size,
        val_tokens=args.val_tokens,
    )

    print(f"encoder        : {meta['encoder_name']} "
          f"(vocab {meta['vocab_size']:,})")
    print(f"shard          : {meta['shard']}")
    for name, loader in (("train", train_loader), ("val", val_loader)):
        s = loader_stats(loader)
        print(f"{name:<5}: tokens {s['tokens']:>11,} | windows {s['windows']:>8,} | "
              f"batches {s['batches']:>6,} | tokens/epoch {s['tokens_per_epoch']:>11,}")

    input_ids, target_ids = next(iter(train_loader))
    print(f"\ninput_ids  shape: {tuple(input_ids.shape)} dtype={input_ids.dtype}")
    print(f"target_ids shape: {tuple(target_ids.shape)}")
    print(f"target == input shifted by one: "
          f"{torch.equal(target_ids[:, :-1], input_ids[:, 1:])}")
    print(f"overlap: stride {train_loader.dataset.stride} vs block_size "
          f"{args.block_size} -> {'none' if train_loader.dataset.stride == args.block_size else 'some'}")
