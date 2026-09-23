"""
--- What is tiktoken? ---
  tiktoken is OpenAI's fast BPE tokenizer written in Rust with Python bindings.
  It does NOT train a new vocabulary; it uses a pre-trained one.
  Under the hood:
    - Text is first split with a regex pattern (handles spaces, punctuation, etc.)
    - Each piece is then encoded byte-by-byte using the pre-trained merge table.
    - The result is a list of integer token IDs.
"""

import os
import tiktoken

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "assets", "llm-book.txt")

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("=" * 60)
print("  BPE TOKENIZER — tiktoken (GPT-2 vocab)")
print("=" * 60)
print(f"Total characters in book: {len(raw_text)}")
print(f"First 99 characters: {repr(raw_text[:99])}\n")

enc = tiktoken.get_encoding("gpt2")

print(f"Vocabulary size (gpt2): {enc.n_vocab}")

token_ids = enc.encode(raw_text, allowed_special="all")
print(f"Total tokens after encoding: {len(token_ids)}")


