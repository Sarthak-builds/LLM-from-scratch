"""Step 1 of training: get WikiText-103 on disk as a flat token-id file.

    python -m llm_from_scratch.data.wikitext_prepare --out-dir data/wikitext

What it does:

1. Downloads `Salesforce/wikitext`, config `wikitext-103-raw-v1` via the
   `datasets` library. The `-raw-v1` config keeps the original surface text;
   the plain `wikitext-103-v1` replaces out-of-vocabulary words with `<unk>`,
   which would quietly poison the token distribution you are trying to learn.
   Splits come pre-separated (train / validation / test) — no manual 90/10 cut
   like the book project needed.

2. Joins each split's non-empty rows with newlines. The raw rows still carry
   WikiText's ` = Section = ` heading markup and blank lines between articles.
   Dropping the *empty* rows is the standard preprocessing; leaving the
   `= Heading =` markup in matches how every published WikiText-103 number was
   produced, so keep it.

3. Encodes with the GPT-2 BPE tokenizer (tiktoken), streaming to disk.

4. Writes `train.bin` / `val.bin` as raw `uint16`, plus `meta.json` with the
   exact token counts.

Why uint16: the largest GPT-2 token id is 50256, which fits in 16 bits. That
halves the file versus int32 (229 MB instead of 458 MB for the train split) and
halves the bytes `np.memmap` has to touch per step.

Why streaming instead of `enc.encode(whole_text)`: the train split is ~500M
characters, which tokenizes to well over 100M ids. A Python list of 130M ints
costs roughly 4-5 GB of RAM, and you then have to keep it alive long enough to
convert it to a numpy array. Encoding a few million characters at a time and
appending each chunk to the file keeps peak RAM in the low hundreds of MB. On a
12 GB Colab VM that is the difference between working and getting OOM-killed.

Run `python -m llm_from_scratch.data.wikitext_prepare --help` for options
(`--shard` builds a small file for smoke tests).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
ENCODER_NAME = "gpt2"
TOKEN_DTYPE = np.uint16
DEFAULT_OUT_DIR = Path("data") / "wikitext"

# Characters per encode chunk. Bigger = fewer Python-level round trips but more
# peak RAM; 2M chars is ~500k tokens, about 4 MB as uint16.
CHUNK_CHARS = 2_000_000
NUM_WORKERS = 8


def join_texts(rows, keep_empty=False):
    """Join WikiText rows into one continuous string.

    The rows arrive as one string per line of the article dump, most of them
    blank separators. Joining the non-blank ones with "\\n" gives a single
    stream with no gaps.
    """
    if keep_empty:
        return "\n".join(rows)
    return "\n".join(t for t in rows if t.strip() != "")


def _encode_chunk(encoder, chunk):
    """encode_ordinary, not encode: a 500M-char corpus must never be scanned for
    special-token markers, and there are none in WikiText anyway."""
    return encoder.encode_ordinary(chunk)


def encode_text_to_file(encoder, text, path, chunk_chars=CHUNK_CHARS,
                        workers=NUM_WORKERS):
    """Tokenize `text`, streaming ids to `path` as uint16. Returns token count.

    tiktoken releases the GIL during encoding, so a thread pool genuinely
    parallelises it — this is the single biggest speedup in the whole script.
    """
    chunks = [text[i: i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        for ids in pool.map(lambda c: _encode_chunk(encoder, c), chunks):
            arr = np.asarray(ids, dtype=TOKEN_DTYPE)
            arr.tofile(fh)
            total += arr.size
            del ids, arr
    tmp.replace(path)                     # atomic: never leave a half .bin
    return total


def load_split_texts(split):
    try:
        from datasets import load_dataset
    except ImportError as exc:                            # pragma: no cover
        raise SystemExit(
            "the `datasets` library is required to fetch WikiText-103.\n"
            "  pip install datasets\n"
            "Run this on the Colab VM, not in the local .venv."
        ) from exc

    print(f"downloading {DATASET_REPO} [{DATASET_CONFIG}] split={split} ...",
          flush=True)
    ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split=split)
    print(f"  rows: {len(ds):,}", flush=True)
    return join_texts(ds["text"])


def prepare(out_dir=DEFAULT_OUT_DIR, splits=("train", "validation"),
            shard=1.0, encoder_name=ENCODER_NAME, keep_empty=False):
    """Download, tokenize and write `<out_dir>/{split}.bin` + `meta.json`."""
    if not 0.0 < shard <= 1.0:
        raise ValueError("shard must be in (0, 1]")

    import tiktoken

    encoder = tiktoken.get_encoding(encoder_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "encoder_name": encoder_name,
        "vocab_size": encoder.n_vocab,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "itemsize": np.dtype(TOKEN_DTYPE).itemsize,
        "shard": shard,
        "keep_empty_rows": keep_empty,
        "splits": {},
    }

    for split in splits:
        name = "train" if split == "train" else "val"
        started = time.time()

        text = load_split_texts(split)
        full_chars = len(text)
        if shard < 1.0:
            text = text[: int(full_chars * shard)]
        print(f"  joined {full_chars:,} chars -> using {len(text):,}", flush=True)

        tokens = encode_text_to_file(encoder, text, out_dir / f"{name}.bin")
        elapsed = time.time() - started
        path = out_dir / f"{name}.bin"
        size_mb = path.stat().st_size / 1024 ** 2

        print(f"  {name}: {tokens:,} tokens -> {path} ({size_mb:,.1f} MB) "
              f"in {elapsed / 60:.1f} min", flush=True)
        meta["splits"][name] = {
            "hf_split": split,
            "file": path.name,
            "tokens": tokens,
            "chars": len(text),
            "bytes": path.stat().st_size,
        }
        del text

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    total = sum(s["tokens"] for s in meta["splits"].values())
    print(f"\ntotal tokens on disk: {total:,}")
    print(f"wrote {out_dir / 'meta.json'}")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"where to write the .bin files (default {DEFAULT_OUT_DIR})")
    parser.add_argument("--shard", type=float, default=1.0,
                        help="fraction of each split to keep; 0.02 makes a ~20 s "
                             "smoke-test dataset, 1.0 is the real thing")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        choices=["train", "validation", "test"])
    parser.add_argument("--encoder", default=ENCODER_NAME)
    parser.add_argument("--keep-empty-rows", action="store_true",
                        help="keep the blank separator rows instead of dropping them")
    args = parser.parse_args(argv)

    prepare(out_dir=args.out_dir, splits=tuple(args.splits), shard=args.shard,
            encoder_name=args.encoder, keep_empty=args.keep_empty_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
